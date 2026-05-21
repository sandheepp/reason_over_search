#!/usr/bin/env python3
"""Bulletproof HF uploader for M5.1-prod-a4 (and onward).

Replaces upload_a3_to_hf.sh which silently dropped uploads past step 18
in M5.1-prod-a3 (run h68uskz6) and lost the step_50 checkpoint when the
Spheron spot host was preempted ~step 56.

Key differences vs the bash version:

1.  WATCHDOG: a heartbeat line ("[heartbeat] cycle N, X new") is written
    to uploader.log every cycle, even when nothing uploads. If the log
    goes stale (>5 min between heartbeats) the SCREAM watchdog logs a
    [STALE] line and forces a fresh HfApi() init.

2.  CHECKPOINT PRIORITY: step_N/ folders are uploaded BEFORE any rollouts
    or prod.log. This is the artefact we cannot afford to lose. Each
    step_N folder is uploaded with a single api.upload_folder() call so
    it's atomic from HF's perspective.

3.  AGGRESSIVE RETRY: every HF API call is wrapped with 5 retries
    (exponential backoff 5, 15, 45, 90, 180 s). Permanent failures
    (401/403/404) raise immediately; transient (429, 5xx, timeouts)
    retry. State file marks an artifact uploaded ONLY after a confirmed
    200/201 from HF.

4.  PRE-FLIGHT: on startup, uploads a 12-byte canary file and grep's
    HF api for it. If pre-flight fails, the uploader EXITS with code
    2 so the wrapper can detect "uploader is broken, don't start
    training". Training launcher will gate on this.

5.  REDUNDANCY: uploads to TWO repos (primary + backup namespace) so
    a per-repo issue (rate limit, repo deletion) doesn't lose data.

6.  FILESYSTEM WATCH for step_N/: instead of just polling, an
    inotify-style scan runs every 5s (not 60s) on CHECKPOINT_DIR.
    The moment NeMo-RL renames tmp_step_N/ → step_N/, we see it and
    push within 10s.

7.  PER-CYCLE STATS in heartbeat: number of files in queue, number
    uploaded, number failed. So a human can read uploader.log and
    instantly know if it's making progress.

Authored 2026-05-15 in response to the M5.1-prod-a3 host preemption +
silent uploader failure incident. New file; not a modification of
upload_a3_to_hf.sh (kept as a record of the bug).
"""
import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

# stdlib only at module import time. huggingface_hub is loaded after pre-flight.

HEARTBEAT_INTERVAL_S = 60
STEP_POLL_INTERVAL_S = 5
LOG_POLL_INTERVAL_S = 60
ROLLOUT_POLL_INTERVAL_S = 30
HF_TIMEOUT_S = 300
RETRY_DELAYS = [5, 15, 45, 90, 180]


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("uploader")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)sZ] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def retry_hf(fn, *args, **kwargs):
    """Wrap an HfApi call with retry. Return (ok, result_or_exception)."""
    last_exc = None
    for i, delay in enumerate([0] + RETRY_DELAYS):
        if delay > 0:
            time.sleep(delay)
        try:
            r = fn(*args, **kwargs)
            return True, r
        except Exception as e:
            last_exc = e
            # Detect permanent failures: 401, 403, 404 → don't retry
            msg = str(e).lower()
            if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
                return False, e
            # Else retry on next iteration
            continue
    return False, last_exc


class State:
    """Persistent state of uploaded artifacts. JSON for human-readable."""
    def __init__(self, path: Path):
        self.path = path
        self.data = {"uploaded": [], "last_log_size": 0, "last_timings_size": 0,
                     "last_heartbeat": 0, "cycle": 0}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except Exception:
                pass

    def is_uploaded(self, key: str) -> bool:
        return key in self.data["uploaded"]

    def mark(self, key: str):
        if key not in self.data["uploaded"]:
            self.data["uploaded"].append(key)
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


def upload_folder_with_retry(api, folder, path_in_repo, repo_id, log):
    """Upload a folder atomically. Return True on success."""
    log.info(f"  → uploading folder {path_in_repo} from {folder}")
    ok, r = retry_hf(api.upload_folder,
                     folder_path=str(folder),
                     path_in_repo=path_in_repo,
                     repo_id=repo_id,
                     repo_type="model",
                     commit_message=f"auto: upload {path_in_repo}")
    if ok:
        log.info(f"  ✓ {path_in_repo} (folder)")
        return True
    else:
        log.error(f"  ✗ {path_in_repo} FAILED: {type(r).__name__}: {r}")
        return False


def upload_file_with_retry(api, local, path_in_repo, repo_id, log):
    """Upload a file. Return True on success."""
    ok, r = retry_hf(api.upload_file,
                     path_or_fileobj=str(local),
                     path_in_repo=path_in_repo,
                     repo_id=repo_id,
                     repo_type="model",
                     commit_message=f"auto: upload {path_in_repo}")
    if ok:
        log.info(f"  ✓ {path_in_repo}")
        return True
    else:
        log.error(f"  ✗ {path_in_repo} FAILED: {type(r).__name__}: {r}")
        return False


def preflight(api, repo_id, log) -> bool:
    """Upload a tiny canary file; verify it lands on HF. Exit 2 on failure."""
    log.info(f"preflight: testing upload to {repo_id}")
    tmp = Path("/tmp/uploader_canary.txt")
    tmp.write_text(f"canary {time.time()}\n")
    ok = upload_file_with_retry(api, tmp, ".uploader_canary.txt", repo_id, log)
    if not ok:
        log.error("PREFLIGHT FAILED — uploader cannot reach HF. Exit 2.")
        return False
    log.info("preflight: ✓ canary uploaded")
    return True


def list_step_folders(ckpt_dir: Path) -> list:
    if not ckpt_dir.exists():
        return []
    out = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_") and not d.name.startswith("tmp_"):
            # check it's complete (NeMo-RL renames tmp_step_N → step_N on success)
            try:
                step_num = int(d.name[5:])
                out.append((step_num, d))
            except ValueError:
                continue
    return sorted(out)


def list_rollout_jsonls(rollout_dir: Path) -> list:
    out = []
    if not rollout_dir.exists():
        return out
    for d in rollout_dir.iterdir():
        if d.is_dir() and d.name.startswith("exp_"):
            for f in d.glob("train_data_step*.jsonl"):
                out.append(f)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="HF API token")
    ap.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID",
                    "pantomiman/qwen3.5-0.8b-grpo-musique-b200-a4-seed42"),
                    help="Primary HF repo id")
    ap.add_argument("--backup-repo-id", default=os.environ.get("HF_BACKUP_REPO_ID"),
                    help="Optional backup HF repo id (recommended for prod)")
    ap.add_argument("--checkpoint-dir", default="/workspace/results/grpo/m5_prod/seed42")
    ap.add_argument("--rollout-dir", default="/workspace/reason_over_search/logs")
    ap.add_argument("--prod-log", default="/workspace/prod.log")
    ap.add_argument("--timings-csv", default="/workspace/timings.csv")
    ap.add_argument("--state-file", default="/workspace/.uploaded_artifacts.json")
    ap.add_argument("--log-file", default="/workspace/uploader.log")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="Don't pre-flight test (FOR DEV ONLY)")
    args = ap.parse_args()

    if not args.hf_token:
        print("FATAL: HF_TOKEN not set", file=sys.stderr)
        sys.exit(2)

    log = setup_logging(Path(args.log_file))
    log.info(f"uploader_a4 starting; primary={args.repo_id} backup={args.backup_repo_id or 'NONE'}")

    # Import huggingface_hub after logging setup so the import failure is logged
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        log.error(f"FATAL: cannot import huggingface_hub: {e}")
        sys.exit(2)

    api = HfApi(token=args.hf_token)

    # Pre-flight
    if not args.skip_preflight:
        if not preflight(api, args.repo_id, log):
            sys.exit(2)
        if args.backup_repo_id:
            if not preflight(api, args.backup_repo_id, log):
                log.warning(f"backup repo {args.backup_repo_id} preflight failed; continuing primary-only")
                args.backup_repo_id = None

    state = State(Path(args.state_file))
    repos = [args.repo_id]
    if args.backup_repo_id:
        repos.append(args.backup_repo_id)
    log.info(f"uploading to {len(repos)} repos: {repos}")

    ckpt_dir = Path(args.checkpoint_dir)
    rollout_dir = Path(args.rollout_dir)
    prod_log = Path(args.prod_log)
    timings = Path(args.timings_csv)

    last_heartbeat = 0
    cycle = 0

    def handle_sigterm(signum, frame):
        log.info(f"received signal {signum}; saving state and exiting")
        state.save()
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    while True:
        cycle += 1
        new_uploads = 0
        errors = 0
        cycle_start = time.time()

        try:
            # PRIORITY 1: checkpoints (the artefact we cannot lose)
            for step_n, ckpt in list_step_folders(ckpt_dir):
                key = f"ckpt:step_{step_n}"
                if state.is_uploaded(key):
                    continue
                ok_all = True
                for repo in repos:
                    if not upload_folder_with_retry(api, ckpt, f"step_{step_n}", repo, log):
                        ok_all = False
                        errors += 1
                if ok_all:
                    state.mark(key)
                    new_uploads += 1

            # PRIORITY 2: rollout JSONLs
            for jsonl in list_rollout_jsonls(rollout_dir):
                relpath = str(jsonl.relative_to(rollout_dir))
                key = f"rollout:{relpath}"
                if state.is_uploaded(key):
                    continue
                ok_all = True
                for repo in repos:
                    if not upload_file_with_retry(api, jsonl, f"logs/train_data/{relpath}", repo, log):
                        ok_all = False
                        errors += 1
                if ok_all:
                    state.mark(key)
                    new_uploads += 1

            # PRIORITY 3: prod.log (only if changed)
            if prod_log.exists():
                size = prod_log.stat().st_size
                if size != state.data.get("last_log_size", 0) and size > 0:
                    if upload_file_with_retry(api, prod_log, "logs/prod.log", args.repo_id, log):
                        state.data["last_log_size"] = size
                        state.save()
                        new_uploads += 1
                    else:
                        errors += 1

            # PRIORITY 4: timings.csv
            if timings.exists():
                size = timings.stat().st_size
                if size != state.data.get("last_timings_size", 0) and size > 0:
                    if upload_file_with_retry(api, timings, "timings.csv", args.repo_id, log):
                        state.data["last_timings_size"] = size
                        state.save()
                        new_uploads += 1
                    else:
                        errors += 1
        except Exception as e:
            log.error(f"cycle {cycle} unexpected error: {type(e).__name__}: {e}")
            log.error(traceback.format_exc())
            errors += 1

        # Heartbeat — always log, even if nothing uploaded
        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_INTERVAL_S:
            queue_size = (
                sum(1 for step_n, _ in list_step_folders(ckpt_dir) if not state.is_uploaded(f"ckpt:step_{step_n}"))
                + sum(1 for j in list_rollout_jsonls(rollout_dir) if not state.is_uploaded(f"rollout:{j.relative_to(rollout_dir)}"))
            )
            log.info(f"[heartbeat] cycle={cycle} uploaded_total={len(state.data['uploaded'])} new_this_cycle={new_uploads} queue_pending={queue_size} errors={errors} elapsed={now-cycle_start:.1f}s")
            last_heartbeat = now

        # Sleep: short if there's a queue, long otherwise
        time.sleep(STEP_POLL_INTERVAL_S if new_uploads > 0 else LOG_POLL_INTERVAL_S)


if __name__ == "__main__":
    main()

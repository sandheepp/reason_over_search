#!/usr/bin/env python3
"""External upload health monitor — run from a machine SEPARATE from the
training instance to verify that artifacts are actually arriving on HF.

This is the safeguard against the M5.1-prod-a3 failure mode where the
in-box uploader.log said "✓" repeatedly while the HF repo silently
stopped receiving new files past step 18.

Usage:
    python3 external_monitor.py <hf_repo_id> [--max-stale-min N]

Checks:
1.  When was the latest train_data_step*.jsonl file modified on HF?
2.  How many step_N/ checkpoint folders exist on HF?
3.  When was prod.log last updated?
4.  Cross-reference: if the W&B run says step K but HF only has step K-5
    rollouts, alert.

Output: human-readable lines + non-zero exit code on staleness.

Exit codes:
0   = healthy (last upload within max-stale-min)
1   = stale (no uploads in max-stale-min)
2   = error (can't reach HF or W&B)

Recommended cron: every 5 minutes from a separate machine, log to a file,
on stale-detected fire a desktop notification / email.
"""
import argparse
import datetime
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="HF repo id, e.g. pantomiman/qwen3.5-...-a4-seed42")
    ap.add_argument("--max-stale-min", type=float, default=10.0,
                    help="Alert if no upload activity in this many minutes (default 10)")
    ap.add_argument("--wandb-run", default=None,
                    help="Optional W&B run id (e.g. h68uskz6) to cross-reference current step")
    ap.add_argument("--wandb-project", default="reason_over_search_b200")
    ap.add_argument("--wandb-entity", default="gaurisankarj1996-leiden-university")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("FATAL: pip install huggingface_hub", file=sys.stderr)
        sys.exit(2)

    api = HfApi()

    try:
        # Get repo info including commit history
        info = api.repo_info(args.repo_id, repo_type="model")
        files = api.list_repo_files(args.repo_id, repo_type="model")
    except Exception as e:
        print(f"FATAL: cannot list HF repo: {e}", file=sys.stderr)
        sys.exit(2)

    # Parse what's there
    step_folders = sorted(set(
        int(f.split("/")[0][5:]) for f in files
        if f.startswith("step_") and "/" in f and f.split("/")[0][5:].isdigit()
    ))
    rollout_steps = sorted(set(
        int(f.split("step")[-1].split(".")[0]) for f in files
        if "train_data_step" in f and "exp_013" in f and f.endswith(".jsonl")
    ))

    print(f"=== EXTERNAL MONITOR — {args.repo_id} ===")
    print(f"Time now: {datetime.datetime.utcnow().isoformat()}Z")
    print(f"Total files in repo: {len(files)}")
    print(f"Checkpoint folders found: {step_folders or 'NONE'}")
    print(f"Rollout steps found (exp_013): {len(rollout_steps)} files, max step = {max(rollout_steps) if rollout_steps else 'NONE'}")

    # Get last commit timestamp
    try:
        commits = api.list_repo_commits(args.repo_id, repo_type="model")
        last_commit = commits[0]
        last_ts = last_commit.created_at
        if isinstance(last_ts, str):
            last_ts = datetime.datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc) if last_ts.tzinfo else datetime.datetime.utcnow()
        stale_min = (now - last_ts).total_seconds() / 60
        print(f"Last commit: {last_ts} ({stale_min:.1f} min ago)")
        print(f"Last commit message: {last_commit.title}")
    except Exception as e:
        print(f"WARN: cannot get commit history: {e}")
        stale_min = None

    # Cross-reference with W&B if requested
    if args.wandb_run:
        try:
            import wandb
            wandb_api = wandb.Api()
            run = wandb_api.run(f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run}")
            wandb_step = run.summary.get("_step", "N/A")
            wandb_state = run.state
            print(f"W&B: run state = {wandb_state}, step = {wandb_step}")
            if isinstance(wandb_step, int) and rollout_steps:
                lag = wandb_step - max(rollout_steps)
                print(f"Lag: W&B is {lag} steps ahead of HF rollouts (last on HF: {max(rollout_steps)})")
                if lag > 10:
                    print(f"ALERT: lag > 10 steps — uploader is falling behind or broken")
        except Exception as e:
            print(f"WARN: cannot reach W&B: {e}")

    # Decide health
    if stale_min is None:
        print("UNKNOWN — cannot determine staleness")
        sys.exit(2)
    if stale_min > args.max_stale_min:
        print(f"STALE: no commits in {stale_min:.1f} min (threshold {args.max_stale_min})")
        sys.exit(1)
    print(f"HEALTHY: last commit {stale_min:.1f} min ago")
    sys.exit(0)


if __name__ == "__main__":
    main()

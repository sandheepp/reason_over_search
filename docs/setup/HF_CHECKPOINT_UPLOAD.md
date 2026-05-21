---
title: HF Hub checkpoint auto-upload — M5.1 production
tags: [setup, m5.1, hf-hub, checkpoint, backup]
source: internal
created: 2026-05-12
updated: 2026-05-12
---

# HF Hub checkpoint auto-upload

Defensive cloud backup for every M5.1 production checkpoint. Decoupled from the training loop so an upload failure cannot crash the run. Added 2026-05-12 in response to the 3-loss postmortems ([`../report/RESULTS_SMOKE_m5.md` §7.8.1](../report/RESULTS_SMOKE_m5.md#781-data-deletion-loss--deleted-a1s-rollout-corpus-during-disk-cleanup-2026-05-11) — same principle as committing rollout jsonls to git, applied to model weights too).

## Mechanism

A separate bash process — `training_m5_1/scripts/upload_ckpts_watcher.sh` — polls the production checkpoint directory every 60 s. When it sees a newly atomic-renamed `step_N/` (NeMo-RL writes `tmp_step_N/` first, then renames to `step_N/` once the save is fully flushed), it uploads the contents to a private HF Hub repo named `${HF_REPO_PREFIX}-m5_${MODE}-seed${SEED}-step${N}`.

The watcher:
- runs in a separate process tree from the training loop;
- never sends signals to or shares file locks with the training process;
- on upload failure, logs and retries on the next poll cycle;
- on its own crash, only stops the *uploader* — training continues uninterrupted;
- skips already-uploaded steps via a state file at `results/grpo/m5_${MODE}/seed${SEED}/.uploaded_steps`.

The state file makes the watcher idempotent across restarts. Killing and relaunching the watcher resumes from the last successfully-uploaded step.

## Configuration

`training_m5_1/.env` (gitignored at `training_m5_1/.gitignore:15`):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
HF_REPO_PREFIX=<username>/qwen3.5-0.8b-grpo-musique
```

The token currently in use is documented in the user's session; rotate it via the HF "Access Tokens" page if it's compromised. The prefix can be any HF account or org the token has *write* access to.

Each ckpt is uploaded to: `${HF_REPO_PREFIX}-m5_${MODE}-seed${SEED}-step${N}`. Example with `HF_REPO_PREFIX=pantomiman/qwen3.5-0.8b-grpo-musique`, mode `prod`, seed `42`, step `50`:

> `pantomiman/qwen3.5-0.8b-grpo-musique-m5_prod-seed42-step50`

A separate repo per checkpoint (not branches/revisions of a single repo) keeps the failure surface small: a bad upload doesn't corrupt prior step uploads, and the Hub UI shows each checkpoint as its own listing with its own README.

## Repo contents per checkpoint

```
${HF_REPO_PREFIX}-m5_<mode>-seed<seed>-step<N>/
├── README.md             # auto-generated; cites step, mode, seed, base model, project
├── weights/              # consolidated safetensors (3.2 GB at save_optimizer=false)
│   ├── shard-00001-model-00001-of-00001.safetensors
│   └── ...
├── tokenizer/            # tokenizer files (~13 MB)
└── training_info.json    # NeMo-RL training metadata
```

## Running the watcher

The watcher is **opt-in** — it does NOT auto-launch with `run.sh`. Operator workflow:

```bash
# Terminal 1 — training (foreground or backgrounded as usual)
bash training_m5_1/scripts/run.sh --mode prod

# Terminal 2 — watcher (background); poll-once-per-minute is default
nohup bash training_m5_1/scripts/upload_ckpts_watcher.sh \
      --mode prod --seed 42 \
    > logs/hf_uploader_m5_prod.nohup.log 2>&1 & disown
echo "uploader pid=$!"
```

Both processes log to their own files; neither writes to the other's log. To stop the watcher cleanly: `kill <pid>` (SIGTERM is handled; the trap logs `watcher stop`).

## Verifying an upload

After the watcher reports `DONE step=50` in `logs/hf_uploader_m5_prod.log`:

```bash
huggingface-cli download "${HF_REPO_PREFIX}-m5_prod-seed42-step50" --repo-type model --local-dir /tmp/_verify
ls /tmp/_verify/weights/  # should show *.safetensors
```

Or visit `https://huggingface.co/${HF_REPO_PREFIX}-m5_prod-seed42-step50` (private; HF login required).

## Why this won't break training

Specific guarantees:

1. **Separate process tree**: training is its own python process under `nohup`; the watcher is its own bash + transient python subprocesses. They share no fds and no signal handlers.
2. **Read-only on the checkpoint side**: the watcher only reads `step_N/` after the atomic rename completes. It never writes inside `step_N/` and never touches `tmp_step_N/` (the in-progress save).
3. **No shared GPU surface**: the watcher only uses CPU + network. No CUDA context; no contention with vLLM's KV cache or DTensor.
4. **Disk pressure**: uploads stream from disk to HF Hub via `huggingface_hub.HfApi.upload_folder` (which uses temporary in-memory chunks, not local copies). The watcher does NOT make local copies of the checkpoint dir.
5. **Bandwidth**: a 3.2 GB checkpoint upload to HF Hub at ~50 MB/s takes ~65 s. On a typical instance with 1+ Gbps network, this doesn't measurably impact training (which is GPU-bound and barely touches the network).
6. **Failure isolation**: `set -uo pipefail` only — NOT `set -e`. The script never propagates a failed upload to `exit`. It logs and continues.

## When to verify the watcher is working

After every production launch:

```bash
# 30 seconds after launch — watcher should have printed the start banner
tail -5 logs/hf_uploader_m5_prod.log
# Expected: "watcher start mode=prod seed=42 ..."

# After first save (step 50 for default save_period): check uploads
tail -10 logs/hf_uploader_m5_prod.log
# Expected: "upload step=50 → ..." then "DONE step=50"
```

If the second check shows `FAIL step=50 rc=...`, inspect the log for a stack trace. Common failures:
- **HF token invalid / no write permission**: regenerate the token at https://huggingface.co/settings/tokens (use "Write" scope).
- **Repo prefix points to an org you don't own**: change `HF_REPO_PREFIX` to your username.
- **Out of disk for HF cache**: `huggingface_hub` caches at `~/.cache/huggingface/`; clear it if disk-pressed.

The watcher retries on the next poll cycle, so a transient network blip resolves on its own.

## What this protects against

| Failure mode | Before this watcher | With this watcher |
|---|---|---|
| `rm -rf` on the checkpoint dir | All saved checkpoints lost | All saved checkpoints recoverable via `huggingface-cli download` |
| Vast instance killed | Checkpoints on `/workspace` lost on shutdown | All uploaded checkpoints survive |
| Disk corruption / fs failure | Local checkpoints lost | All uploaded checkpoints survive |
| Network outage during upload | n/a | Watcher logs failure, retries next poll cycle |
| Watcher crash | n/a | Training unaffected; restart watcher, state file resumes from last good step |

## What this does NOT protect against

- **Rollout jsonl files** (the `train_data_step*.jsonl` in `logs/exp_NNN/`): the watcher only uploads checkpoint weights. For rollout-corpus backup, follow rule #5 of [`RESULTS_SMOKE_m5.md §7.8.1`](../report/RESULTS_SMOKE_m5.md#781-data-deletion-loss--deleted-a1s-rollout-corpus-during-disk-cleanup-2026-05-11): commit them to git after each ckpt save.
- **W&B run artifacts** (scalar metrics, system metrics): those are uploaded to W&B by the training process directly. W&B run loss is a separate concern from local-disk loss.

## Pointers

- Watcher script: [`../../training_m5_1/scripts/upload_ckpts_watcher.sh`](../../training_m5_1/scripts/upload_ckpts_watcher.sh)
- Env template: [`../../training_m5_1/.env.example`](../../training_m5_1/.env.example)
- Origin postmortem (why this rule exists): [`../report/RESULTS_SMOKE_m5.md` §7.8.1](../report/RESULTS_SMOKE_m5.md#781-data-deletion-loss--deleted-a1s-rollout-corpus-during-disk-cleanup-2026-05-11)

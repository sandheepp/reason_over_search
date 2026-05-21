---
title: Cadence Handoff — Step-by-step for running 10-step cadence cycles on M5.5 H200
tags: [m5_5, h200, ops, cadence, handoff]
source: internal
created: 2026-05-17
updated: 2026-05-17
---

# Cadence Handoff (M5.5 H200)

End-to-end recipe for one cadence-update cycle on the **M5.5 H200** training run (Qwen3.5-0.8B GRPO on MuSiQue, **F1 + 0.1 floor + format-gate reward**). Hand this to another agent and they should take over without context.

Companion to [`docs/report/RESULTS_M5_5_H200.md`](../report/RESULTS_M5_5_H200.md) (the canonical narrative), and modelled after [`docs/milestone_5/CADENCE_HANDOFF.md`](CADENCE_HANDOFF.md) (the M5.1-H200 sibling). Differences from that doc:

- **No Docker.** This run lives directly on a bare-host Spheron H200 (no `h200-a4` container). Use `sudo` on the host instead of `docker exec`.
- **Persistent volume** is `/mnt/milestone55` (virtiofs, 500 GB). Checkpoints write directly to it via a symlink at `/root/reason_over_search/results/grpo/m5_5_prod_h200/seed42`.
- **One analyzer script** does the work: [`analyze_rollouts.py`](../../training_m5_1/scripts/analyze_rollouts.py) (reused from M5.1; no M5.5-specific changes needed). Output is the verbatim block that goes into the RESULTS doc.
- **No HF README separate file maintained.** The HF repo's `README.md` was pinned at backfill time and isn't synced per-cadence (the internal RESULTS doc carries the narrative; HF visitors see a stable description). If you want to add cadence rows to HF, do it manually.

## Quick context

The training run is GRPO on Qwen3.5-0.8B / MuSiQue / **`max(0.1, F1)` if `<answer>` else `0` reward** (the M5.5 ablation knob: F1 + 0.1 partial-credit floor + format-gate), running on Spheron H200 with persistent virtiofs volume `milestone55`. Every 10 GRPO steps a checkpoint lands at `/mnt/milestone55/ckpts/m5_5_prod_h200/seed42/step_N/`. The `hf_poller` tmux session pushes new ckpts + rollouts + logs to [`sandheepp/qwen3.5-0.8b-grpo-musique-h200-m5_5-seed42-f1-floor-fmt`](https://huggingface.co/sandheepp/qwen3.5-0.8b-grpo-musique-h200-m5_5-seed42-f1-floor-fmt) (public, single repo with subdirs). The cadence cycle = pull the new step's data, write an analysis block, append to RESULTS, commit, push GitHub.

Main doc being updated: [`docs/report/RESULTS_M5_5_H200.md`](../report/RESULTS_M5_5_H200.md). Currently fires every ~40-60 min depending on step wall time.

## Connection setup (one-time per session)

Get the current SSH endpoint from the user (Spheron hosts rotate on preemption; as of 2026-05-17 this was `ubuntu@204.12.171.143`).

```bash
# Test connection — should print today's date
ssh -o StrictHostKeyChecking=no ubuntu@<host> 'date -u'

# Confirm the three required tmux sessions are alive
ssh ubuntu@<host> 'sudo tmux ls'
# Expected: train, sync_to_volume, hf_poller (and optionally watch_resources)
```

If `train` is missing the run has died — escalate to the user before re-launching. If `hf_poller` is missing, see [§Recovery](#recovery) below.

## The analyzer

One script does all the work: [`analyze_rollouts.py`](../../training_m5_1/scripts/analyze_rollouts.py) (committed in M5.1 branch; lives at `/tmp/analyze_rollouts.py` on the M5.5 H200 box, mirrored to `/mnt/milestone55/cadences/` for persistence).

```bash
# Standard cadence: steps S to S+9
ssh ubuntu@<host> 'sudo python3 /tmp/analyze_rollouts.py /root/reason_over_search/logs/exp_013 S E'
# Where S=cadence_start, E=cadence_end (e.g. 101 and 110 for cadence 11)
```

Output is Markdown ready to paste under `### Cadence N — steps S-E` in the RESULTS doc. Includes:
- **Window aggregate**: reward mean / std / max / %zero / %nonzero, turns p50/p95/max, tool calls p50/p95/max, completion% (% with `</answer>`), truncation%, response char length p50/p95, input length mean
- **Per-step trajectory**: n, reward mean, reward max, % with answer, avg turns, avg tools — one row per step
- **5 example trajectories** from `sample_step` (default = end step): mix of highest-reward and zero-reward rollouts, with truncated content + the assistant's last chunk

For deeper analysis (multi-hop wins, planned-multi-hop, silent-flip rate), the M5.1-H200 cadence cycle has 6 more specialized scripts ([`docs/milestone_5/CADENCE_HANDOFF.md`](CADENCE_HANDOFF.md) §"The 7 scripts that do all the work"). Those scripts were not committed (they live in `/tmp/` on the M5.1 box). If you need them for M5.5, port them by adapting paths: `m5_prod`/`exp_032`/`/workspace/...` → `m5_5_prod_h200`/`exp_013`/`/root/reason_over_search/...`.

## Step-by-step cadence cycle

### Step 1 — Detect that a new ckpt has landed

The trigger is `step_N0` checkpoint directory appearing under `/mnt/milestone55/ckpts/m5_5_prod_h200/seed42/`. Tip: `ls | sort -V` sorts numerically (without `-V`, `step_100` lexically precedes `step_20`).

```bash
ssh ubuntu@<host> 'sudo bash -c "
date -u
ls /mnt/milestone55/ckpts/m5_5_prod_h200/seed42 | sort -V | tail -5
ls -lt /root/reason_over_search/logs/exp_013/train_data_step*.jsonl 2>/dev/null | head -3
grep -E '^=========================' /root/logs/m5_5_chain_seed42_20260516T2227Z.log | tail -3
"'
```

If `step_N0` is not yet listed: the in-flight step's wall time hasn't elapsed. Compute ETA from the latest `Total step time` lines in `chain log` and re-arm a wakeup for that ETA + ~5 min buffer.

A Monitor-based watcher works too (see [§Wakeup automation](#wakeup-automation)).

### Step 2 — Pull the analysis data

```bash
ssh ubuntu@<host> 'sudo python3 /tmp/analyze_rollouts.py /root/reason_over_search/logs/exp_013 <START> <END> 2>&1 | tee /mnt/milestone55/cadences/cadence_$(printf %02d $((START/10+1))).md'
# Then scp it locally:
scp 'ubuntu@<host>:/mnt/milestone55/cadences/cadence_NN.md' /tmp/
```

The key numbers per cadence (extract while pasting into RESULTS):
- **reward mean** (the headline number)
- **% nonzero** (watch the floor-vs-real-F1 split — see [reward design](../report/RESULTS_M5_5_H200.md#6-reward-design-the-ablation-knob))
- **mean tool calls** (watch for over-search drift up, or compression drift down)
- **completion %** (should stay 100% after cadence 2)
- **median response chars** (watch for re-expansion toward 8 K truncation cap)

Headline pattern for M5.5: **reward and tool-call count are inversely correlated** during efficiency-discovery cadences (model gets same/higher reward with fewer tool calls). Drift cycles look like: tool calls peak → reward gain stalls → model corrects down → next cadence reward climbs again. Cadence 6→8 is the textbook example so far.

### Step 3 — Write the cadence section

Open [`docs/report/RESULTS_M5_5_H200.md`](../report/RESULTS_M5_5_H200.md). Find `### Cadence N — steps X-Y` for the most recent cadence (currently 9 ending at step 90). Insert the new cadence right after it, before `## 9. Cost / wall-clock estimate`.

Update the cadence-summary table at the top of §7 with the new row. If the new cadence sets a new reward high, mention it in the run header.

**Standard section template** (matches cadences 1-9):

```markdown
### Cadence N — steps X-Y

<verbatim analyzer output from /mnt/milestone55/cadences/cadence_NN.md>

**Commentary (cadence N):** <one paragraph; what's the headline story>
- reward direction (vs prior cadence)
- tool-call direction (vs prior cadence) — is this over-search drift or compression?
- completion / truncation direction
- step time direction
- any single-step run-highs in this window
- watch item for next cadence
```

For cadences where something notable happens (drift cycle peaks, reward plateau, F1=1.0 spike), expand the commentary to 2-3 paragraphs. The M5.1-H200 doc has good models in C6 (over-search peak) and C8 (efficiency win).

### Step 4 — Commit + push GitHub

```bash
cd /Users/sandheepp/broadsword/reason_over_search
git add docs/report/RESULTS_M5_5_H200.md
git commit -m "$(cat <<'EOF'
docs(m5_5/h200): cadence N (steps X-Y) - <one-line summary>

<2-3 line key finding>
EOF
)"
git push origin m5.5
```

**Branch is `m5.5`** (not `m5.5_h200` despite the file name in some places).

### Step 5 — HF repo

The `hf_poller` tmux session on the H200 box auto-uploads new step folders, rollout jsonls, and the 4 log files every 60s. **You don't need to do anything for HF in the normal cadence cycle.** Verify it's still alive once per cadence:

```bash
ssh ubuntu@<host> 'sudo tmux ls | grep hf_poller && sudo tail -3 /mnt/milestone55/logs/hf_poller.log'
```

The HF `README.md` was pinned at backfill time and is NOT synced per cadence — keeps the visitor-facing description stable. If you want to update it, edit manually and push to HF separately (see [recovery section](#recovery) for the auth pattern).

### Step 6 — Re-arm the wakeup

Use the ScheduleWakeup tool with:
- `delaySeconds`: ETA of next `step_(N+1)0` based on the latest step wall time (typically 9 × current_step_wall + ~5 min buffer). Cap is 3600 s.
- `prompt`: `<<autonomous-loop-dynamic>>`
- `reason`: one-sentence explaining what cadence you're waking for and what to watch

If the next ckpt won't land before the 3600 s cap (step wall > 400 s), schedule for 3600 and accept a brief re-arm on wake. Alternatively use a Monitor on the chain log for the `Step (N+1)0/311` boundary — more efficient than polling, and the same Monitor catches `Saving checkpoint` events.

## Sources of truth for current numbers

- **Step time per step**: `grep "Total step time" /root/logs/m5_5_chain_seed42_20260516T2227Z.log` (host, not container). Each line is one step in order.
- **Reward per step**: `analyze_rollouts.py` output (reads jsonl; authoritative).
- **W&B run**: [`gtf8xe1d` in project `reason_over_search_b300_m5_5`](https://wandb.ai/sandheep-p01-medichive/reason_over_search_b300_m5_5/runs/gtf8xe1d). Use only if you don't have host access. The jsonl + chain log are authoritative. (Project name has leftover `_b300_` from the b300-derived config — not renamed mid-run.)
- **HF poller state**: `/mnt/milestone55/logs/hf_poller_state.json` — tracks which ckpts and rollouts were already pushed. If something seems missing on HF, check this file first.
- **Volume contents (cadences + ckpts + logs)**: `/mnt/milestone55/` (also visible from a fresh box if you re-mount the volume).

## Common gotchas

1. **`ls | sort` puts `step_100` before `step_20`** (lexical). Always use `sort -V` for numeric sort on step dirs.
2. **`/tmp/analyze_rollouts.py` may be wiped on container restart** (the H200 host doesn't actually restart often, but if it does, `scp` the script back from `training_m5_1/scripts/analyze_rollouts.py` in this repo).
3. **The `exp_013/` rollout dir name is hardcoded** in cadence scripts and in this handoff. If NeMo-RL bumps to `exp_014/` on a re-launch, update the paths.
4. **`extract_solution` finds the FIRST `<answer>` block**, but the system prompt also contains an example `<answer> Beijing </answer>`. Custom regex scrapes must use `re.findall(...)[-1]` to get the LAST match (the model's actual answer). The `analyze_rollouts.py` analyzer already handles this correctly.
5. **Tool-call count includes 2 system-prompt mentions** of `<tool_call>` in the tools schema. If you write a custom counter, subtract: `actual_tools = max(0, txt.count("<tool_call>") - 2)`. `analyze_rollouts.py`'s `tools` field already handles this.
6. **Reward field is per-token, not per-rollout.** Each rollout has `rewards: [float]` — a list, one float per generated token. The final scalar is `max(rewards)` (which is what `analyze_rollouts.py` uses). Naive `rewards[-1]` gives 0.0 for most rollouts because the last token is padding.
7. **The 0.1 floor is per-rollout** in the reward histogram. Cadences with `% nonzero ≈ 100%` aren't healthy *unless* you also check the spread above the floor (`reward > 0.105`). A model parking at the floor will have %nonzero = 99% but mean ≈ 0.10. Always check **mean reward** alongside **% nonzero**.
8. **Re-uploading checkpoints to HF**: the `hf_poller_state.json` tracks pushed steps; if you want to re-push (e.g. you accidentally deleted a step from HF), edit the state file to remove that step from the `ckpts` list, then the next poll cycle re-uploads it.
9. **Symlink-routed checkpoint dir**: `/root/reason_over_search/results/grpo/m5_5_prod_h200/seed42` is a symlink to `/mnt/milestone55/ckpts/m5_5_prod_h200/seed42`. Don't `rm -rf` the local path expecting to only delete the symlink — the `-rf` follows symlinks. Use `rm` (no `-rf`) on the symlink itself.

## Wakeup automation

For unattended cadence cycles, prefer Monitor over ScheduleWakeup:

```text
Monitor(
  description="prod_h200 step + ckpt boundaries (cadence triggers)",
  timeout_ms=3600000,
  command='ssh ubuntu@<host> "sudo tail -F /root/logs/m5_5_chain_seed42_20260516T2227Z.log 2>/dev/null"'
          ' | grep -aE --line-buffered "(=+ Step (10[05]|1[1-9][05]|2[0-9][05]|3(00|10|11))/311'
          '|Saving checkpoint for step|Traceback|RuntimeError|OutOfMemoryError|Killed)"'
)
```

Adjust the step-number alternation to match the current cadence range. Every 5th step boundary keeps notifications sparse; checkpoint saves fire every 10 steps. Monitor times out after 1 h — re-arm on next wake.

## Decision points the user typically owns

These are not autonomous; surface them to the user:
- **Early-stop call**: when to declare the run done before step 311. Currently the user wants to go to step 311.
- **Host preemption / migration**: when to switch instances. (Persistent volume + HF backup makes this near-zero-loss; ~10 min for re-bootstrap on fresh box.)
- **Reward extension prototype**: any tool-call bonus or chain-quality bonus would be a new ablation run (M5.6+), not a mid-run modification.
- **HF repo housekeeping**: ckpt retention policy (currently: keep all forever; HF storage is free at this scale).

## What an agent should NOT do without asking

- Don't modify `training_m5_5/configs/m5_5_research_paper_h200.yaml` or any training-side code while the run is live.
- Don't touch the reward function `training_m5_5/src/rewards/search_r1.py` — that's the M5.5 spec being tested.
- Don't force-push to `m5.5` (`git push --force`); use regular pushes.
- Don't kill the `train`, `sync_to_volume`, or `hf_poller` tmux sessions without checking with user first.
- Don't change `CHECKPOINT_DIR_BASE` in `.env` while training is running — NeMo-RL reads it at launch; mid-run changes leave new ckpts in the old path.
- Don't `rm -rf` the symlinked local seed42 path (see [gotcha #9](#common-gotchas)).

## Recovery

### `hf_poller` died (visible in `tmux ls`)

```bash
ssh ubuntu@<host> 'sudo tail -30 /mnt/milestone55/logs/hf_poller.log'
# Diagnose, fix, then relaunch:
ssh ubuntu@<host> 'sudo tmux new-session -d -s hf_poller "bash -c \"set -a; . /root/reason_over_search/training_m5_5/.env; set +a; /root/reason_over_search/training_m5_5/nemo_rl/.venv/bin/python /tmp/m5_5_hf_poller.py 2>&1 | tee -a /mnt/milestone55/logs/hf_poller.log\""'
```

The poller is idempotent — state file ensures no re-uploads.

### Instance preempted (lost SSH)

The persistent volume survives. On the replacement box:
1. Mount: `sudo mount -t virtiofs milestone55 /mnt/milestone55`
2. Re-bootstrap: `bash training_m5_5/scripts/start_b300.sh --smoke-first --mode prod_h200`
3. Set `CHECKPOINT_DIR_BASE=/mnt/milestone55/ckpts` in `training_m5_5/.env` **before** launching
4. Resume via `checkpointing.resume_from_checkpoint=latest` Hydra override

Full recipe: [`docs/setup/B300_RUNBOOK.md` §Persistent volume](../setup/B300_RUNBOOK.md#persistent-volume-spheron--virtiofs--survive-instance-termination).

### HF push fails with "fetch first" (race with poller)

Wait 60s and retry. The poller commits via the HfApi (not git), so explicit `git push` clones from `/tmp/hf-readme-sync/` (if you're updating the HF README manually) race against poller commits. Re-clone with `GIT_LFS_SKIP_SMUDGE=1` to avoid pulling 100+ GB of safetensors:

```bash
cd /tmp && rm -rf hf-readme-sync
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/sandheepp/qwen3.5-0.8b-grpo-musique-h200-m5_5-seed42-f1-floor-fmt hf-readme-sync
# edit README.md, then:
cd /tmp/hf-readme-sync && git add README.md && git commit -m "..."
HF_TOK=$(grep ^HF_TOKEN= ~/path/to/.env | cut -d= -f2)
git push "https://sandheepp:${HF_TOK}@huggingface.co/sandheepp/qwen3.5-0.8b-grpo-musique-h200-m5_5-seed42-f1-floor-fmt" HEAD:main
```

## Reference: cadence trajectory through cadence 9

(Updated 2026-05-17; live in [`docs/report/RESULTS_M5_5_H200.md` §7](../report/RESULTS_M5_5_H200.md#7-cadence-summary-steps-190).)

| # | Steps | rew_mean | % nonzero | mean tool calls | completion% | step wall (s) |
|---|---|---:|---:|---:|---:|---:|
| C1 | 1-10 | 0.110 | 56% | 3.88 | 68% | ~600-1200 (cold) |
| C2 | 11-20 | 0.160 | 99% | 1.15 | 100% | ~130-200 |
| C3 | 21-30 | 0.189 | 100% | 1.10 | 100% | ~130-150 |
| C4 | 31-40 | 0.221 | 99% | 2.55 | 100% | ~250-380 |
| C5 | 41-50 | 0.233 | 99% | 2.60 | 100% | ~320-450 |
| C6 | 51-60 | 0.258 | 99% | **4.30** | 99% | ~440-1085 (over-search peak) |
| C7 | 61-70 | 0.258 | 100% | 3.38 | 100% | ~270-440 |
| C8 | 71-80 | 0.269 | 100% | 2.14 | 100% | ~240-290 |
| C9 | 81-90 | **0.290** | 100% | 2.21 | 100% | ~250-290 |

**Headline story so far**: reward `0.110 → 0.290` (+164%) while mean tool calls drifted `3.88 → 2.21` (-43%). Better answers with less search. No tool-collapse despite the 0.1 floor.

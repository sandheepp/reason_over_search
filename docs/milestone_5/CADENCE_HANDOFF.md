---
title: Cadence Handoff — Step-by-step for running 10-step cadence cycles on M5.1 H200 a4
tags: [m5_1, h200, ops, cadence, handoff]
source: internal
created: 2026-05-17
updated: 2026-05-17
status: PAUSED at step_180 (run held; resume path documented below)
---

# Cadence Handoff

End-to-end recipe for running one cadence-update cycle on the M5.1 H200 a4 training run. Hand this to another agent and they should be able to take over without context.

## ⏸ STATUS: PAUSED at step_180 (2026-05-17)

The cadence loop is **on hold** at step_180. Not crashed; deliberately held while M8.2 is scoped (decision rationale in [`../report/RESULTS_M5_1_H200.md` §9.6](../report/RESULTS_M5_1_H200.md)). Background:

- step_180 ckpt landed 2026-05-17 ~08:18 UTC (last upload event in `state/uploader_prod.log`).
- The previous Dedicated host went down shortly after.
- Four consecutive Spot replacements (`204.12.168.156` → `204.12.168.241` → `204.12.170.203` → `204.12.171.221`) all dropped SSH mid-bring-up, each preempted before the ~10-15 min docker pull could finish. Pattern: Spot tier is currently churning faster than the bring-up sequence (mount → pull → patch → retriever → train).
- Persistent volume `miletone5` preserved all state across every preemption. No data lost.

**Don't re-run cadences without checking with the user first.** The next experiment is M8.2 (chain-quality reward, [`../milestone_8/MILESTONE_8.md`](../milestone_8/MILESTONE_8.md)), not "more M5.1 steps".

### If user says "resume M5.1"

1. **Use Dedicated tier** (`$4.70/h`), not Spot. Four preemptions in a row says Spot can't fit the ~25 min bring-up window today.
2. Mount + fstab the volume (see "Mount + bring-up" below).
3. `sudo docker pull pantomiman/reason-over-search-v1:v2` (~15 min).
4. Start container per the original recipe (sed FlashInfer GDN patch on BOTH vLLM venvs — see Gotcha #6).
5. Bring up retriever + uploader.
6. Launch with W&B resume: `WANDB_RUN_ID=fde3cib7 WANDB_RESUME=allow bash training_m5_1/scripts/run.sh`. Same W&B run continues.
7. First new step (181) lands → resume cadence loop at C19 (steps 181-190).

## Quick context

The training run is GRPO on Qwen3.5-0.8B / MuSiQue / F1-only reward, running on Spheron with a persistent volume `miletone5`. Every 10 GRPO steps a checkpoint lands. The uploader pushes everything to HF [`pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only). The cadence cycle = pull the new step's data, write an analysis block, commit to GH, sync to HF. Currently fires every ~50-90 min depending on step wall time.

Main doc being updated: [`../report/RESULTS_M5_1_H200.md`](../report/RESULTS_M5_1_H200.md). HF README is maintained separately (cleaner / shorter).

## Connection setup (one-time per session)

The host changes (Spheron preemption + host migrations). Get the current SSH endpoint from the user. As of 2026-05-17 09:00 UTC the active host was `ubuntu@204.12.171.126` (Dedicated, switching to a new Spot). The container inside it is named `h200-a4`.

```bash
# Test connection — should print today's date
ssh -o StrictHostKeyChecking=no ubuntu@<host> 'docker exec h200-a4 date -u'
```

If the container name has changed, list running containers with `ssh ... 'docker ps'`.

## The 7 scripts that do all the work

All in `/tmp/` inside the container. They survive container restarts because `/tmp/` is bind-mounted from the host. If any are missing, the source is in this commit history; re-copy via `scp ... && docker cp`.

| Script | Purpose | Args |
|---|---|---|
| `/tmp/reward_trend2.py` | Per-step `rew_mean / rew>0% / perfect / tool_med / len_med` | start_step end_step |
| `/tmp/cadence_compact.py` | BEST / WORST / MEAN rollout with question + answer + last `<think>` block | start_step end_step |
| `/tmp/hop_strat.py` | Hop-stratified BEST (1 / 2 / 3 / 4+ hops) | start_step end_step |
| `/tmp/planned.py` | Count of planned-multi-hop rollouts (≥3 tool calls + explicit numbered plan) + top plan_score traces | start_step end_step |
| `/tmp/chain_audit.py` | Silent-flip detector → flip rate among reward ≥ 0.9 rollouts | start_step end_step |
| `/tmp/verify_one.py` | Full text of one specific rollout (use for deep-dive on a single trace) | step idx |
| `/tmp/show4hop.py` | Filter for the BEST 4-hop+ trace in a single step | step |

Standard cadence runs the first 5 in batch:

```bash
ssh ubuntu@<host> 'docker exec h200-a4 bash -c "
for script in reward_trend2 cadence_compact hop_strat planned chain_audit; do
  echo --- \$script ---
  /venv/main/bin/python /tmp/\$script.py <START> <END> 2>&1 | head -50
done
"'
```

For a cadence covering steps 171-180: `START=171 END=180`.

## Step-by-step cadence cycle

### Step 1 — Detect that a new ckpt has landed

The trigger is `step_N0` checkpoint dir appearing under `/workspace/reason_over_search/results/grpo/m5_prod/seed42/`. Tip: `ls | sort -V` sorts numerically (without `-V`, `step_100` lexically precedes `step_20`).

```bash
ssh ubuntu@<host> 'docker exec h200-a4 bash -c "
date -u
ls /workspace/reason_over_search/results/grpo/m5_prod/seed42/ 2>/dev/null | sort -V | tail -5
ls -lt /workspace/reason_over_search/logs/exp_032/train_data_step*.jsonl 2>/dev/null | head -3
grep -E \"^=========================\" /workspace/prod.log 2>/dev/null | tail -3
"'
```

If `step_N0` is not yet listed: the in-flight step's wall time hasn't elapsed. Compute ETA from the latest `Total step time` lines in `prod.log` and re-arm a wakeup for that ETA + ~5 min buffer.

### Step 2 — Pull the analysis data

Run the 5 scripts above for the cadence's step range. This typically takes 30-60 s end-to-end (the chain_audit one scans 3,200 rollouts per cadence).

Capture the output. The key numbers per cadence:

- **rew_mean** per step + **window mean** (the headline reward number)
- **tool_med** per step (watch for drift up or down vs prior cadence)
- **len_med** per step (watch for context growth vs the 8 K-token cap)
- **rew > 0 %** per step + window
- **perfect** count per step (rollouts at reward ≥ 0.9)
- **Step wall time** per step (from `grep "Total step time" /workspace/prod.log | tail -12`)
- **BEST / WORST / MEAN** rollout: extract question, final answer, last `<think>`. Comment on each.
- **Hop-stratified BEST**: 1 / 2 / 3 / 4+ hop best examples
- **Planned-multi-hop count** + top plan_score rollout
- **Chain-flip rate**: % of reward ≥ 0.9 rollouts with silent entity flips. Compare to prior cadences.

### Step 3 — Write the cadence section

Open [`docs/report/RESULTS_M5_1_H200.md`](./RESULTS_M5_1_H200.md). Find the most recent `### Cadence N:` section and append the new one right after it (before `## 9. Cost / wall-clock estimate`).

The standard section template:

```markdown
### Cadence N: steps X-Y (one-line characterisation)

| Step | Wall (s) | M:S | rew mean | rew > 0 | tool_med | len_med | notes |
|---:|---:|---:|---:|---:|---:|---:|---|
| <per-step rows, 10 of them> |

**Cadence N vs prior windows**:
| Window | rew_mean | rew > 0 | step wall | tool_med | len_med | 4-hop+ wins | planned-3-5 | flip-rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <last 3-4 cadences> |
| **N (this)** | ... |
| Δ vs prior | ... |

**Trends after cadence N**: (one paragraph; what's the headline story)
- Reward direction
- Tool / len / wall direction
- Flip rate direction
- HF: `step_N0/` live at ...
- Cumulative cost
- ETA + revised projection

#### Mechanical examples (cadence N)

**BEST** — step ..., sample ..., **reward 1.000**, ... tool calls, ... K chars
> Q: ...
> Final answer: `...` ✓
> Commentary: ...

**WORST** — step ..., sample ..., **reward 0.000**, ...
> Q: ...
> Final answer: `...` ✗
> Commentary: ...

**MEAN** — step ..., sample ..., **reward 0.2X**, ...
> Q: ...
> Final answer: ...
> Commentary: ...

#### Claude hand-analyses (cadence N)

1. (Headline finding — what changed mechanically and what it implies)
2. (Secondary finding — usually about flip-rate / cost-vs-reward / capability)

#### Hop-stratified BEST successes (cadence N)

| Hops | Step | Tools | Answer | Question |
|---:|---:|---:|---|---|
| 1 | ... |
| 2 | ... |
| 3 | ... |
| 4+ | ... |

(One sentence about 4-hop+ count vs prior cadences.)

#### Planned-multi-hop reasoning (cadence N)

**X rollouts** with explicit numbered plan + reward 1.0. Top plan_score: step ... sample ... (plan_score Y, Z calls). (One sentence about the trend.)

#### Cadence-N summary

(One paragraph: reward number, structural costs, pattern, watch item for next cadence.)
```

**Style rules** (carry from CLAUDE.md):
- No em-dashes (`—`, `--`); use semicolons, colons, parentheses, or "X to Y"
- No emojis
- Cite paper sections by URL when relevant
- Single-step run-highs go in the run-highs table near the top of §8

### Step 4 — Commit + push GH

```bash
cd /Users/somedude/Documents/Obsidian/code/omega/reason_over_search
git add docs/report/RESULTS_M5_1_H200.md
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
docs(m5_1/h200): cadence N (steps X-Y) - <one-line summary>

<2-3 line key finding>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git push origin experiment_1_h200
```

### Step 5 — Sync HF README

The HF README is **separate** from the internal RESULTS doc. It's a clean self-contained doc for HF visitors. Update only:
1. The status line at the top (step / 311 + UTC time)
2. The `step_10/`, ..., `step_N0/` range in the "What's in this repo" table
3. The `STEP = "step_N0"` line in the quickstart code block
4. The new cadence row in the trajectory table
5. New single-step run-highs in the run-highs table (only if rew_mean ≥ 0.29 to keep the table tight)

The trajectory table on HF deliberately does NOT have per-cadence BEST/WORST/MEAN traces (those are internal noise). Keep HF terse.

```bash
cd /tmp && rm -rf hf-readme-sync
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only hf-readme-sync
# wait for clone
# Edit /tmp/hf-readme-sync/README.md (the 5 items above)
cd /tmp/hf-readme-sync
git add README.md
git -c user.email=gaurisankarj1996@gmail.com -c user.name=Sankar -c commit.gpgsign=false commit -m "docs: cadence N — <one-line summary>" -- README.md
HF_TOK=$(cat ~/.cache/huggingface/token)
git push "https://pantomiman:${HF_TOK}@huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only" HEAD:main
```

`GIT_LFS_SKIP_SMUDGE=1` is **mandatory**; without it the clone tries to pull 100+ GB of safetensors.

If push fails with "fetch first", re-clone (the uploader pushed checkpoint files in parallel). Race-loss recovery: re-clone, re-apply the edits, re-push.

### Step 6 — Re-arm the wakeup

Use the ScheduleWakeup tool with:
- `delaySeconds`: ETA of next `step_(N+1)0` based on the latest step wall time (typically 9 × current_step_wall + ~5 min buffer). Cap is 3600 s; if more than 1 h, use 3600 and accept that you'll need a brief re-arm.
- `prompt`: `<<autonomous-loop-dynamic>>`
- `reason`: one-sentence explaining what cadence you're waking for and what to watch

If the next step won't land before the 3600 s cap (e.g. step wall is 600+ s), schedule for 3600, then on wake do a short re-arm (300-900 s) to catch the actual landing.

## Sources of truth for current numbers

- **Step time per step**: `grep "Total step time" /workspace/prod.log | tail -<N>` on the host. Each line is one step in order.
- **Reward per step**: `reward_trend2.py` output (most reliable; reads jsonl).
- **W&B run**: `fde3cib7` in project `reason_over_search_h200` ([dashboard](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_h200/runs/fde3cib7)). Use only if you don't have container access; the jsonl + prod.log are authoritative.
- **HF uploader state**: `/workspace/state/.uploaded_artifacts.json` inside the container — tracks what's been pushed. If something seems missing from HF, check this file first.

## Common gotchas

1. **HF API needs auth even for public read** of the `tree/main` endpoint. Use `curl -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)"`. Public `resolve/` paths work without auth.
2. **`ls | sort` puts `step_100` before `step_20`** (lexical). Always use `sort -V` for numeric sort on step dirs.
3. **`/tmp/` scripts may be wiped on container restart**. The scripts are in this repo's commit history; re-copy via `scp /local/script.py ubuntu@host:/tmp/ && ssh ... 'docker cp /tmp/script.py h200-a4:/tmp/'`.
4. **The `cadence_compact.py` script reads from `exp_030` by default**; if rollouts are now in a new exp dir (e.g. `exp_033` after a host migration), patch the `load_step()` function or add the new dir to its fallback list. Current code checks both `exp_030` and `exp_032`.
5. **Per-rollout `extract_solution` finds the FIRST `<answer>` block, but the system prompt also contains an example `<answer> Beijing </answer>`**. Scripts using regex must filter — use `re.findall(...)[-1]` to get the LAST match (the model's actual answer).
6. **Tool-call count includes 2 system-prompt mentions**. Subtract: `actual_tools = max(0, txt.count("<tool_call>") - 2)`.
7. **The HF README must NOT be a verbatim mirror of the internal RESULTS doc** — keep HF clean (~200 lines, no incident chains, no cross-run comparisons). The user asked for this explicitly; honour it.
8. **HF README race condition**: the bulletproof uploader pushes to HF continuously. If the README push fails with "fetch first", re-clone with `GIT_LFS_SKIP_SMUDGE=1` and re-push.
9. **vLLM + FlashInfer GDN deadlock on Hopper**: every fresh container (e.g. after a host switch) needs the `qwen3_next.py:156` patch re-applied to both sync and async vLLM venvs. See [`docs/spheron/SETUP_SPHERON.md` §9.1](../spheron/SETUP_SPHERON.md). If a fresh container hangs at step 1 for 60+ min, this is the cause.

## Decision points the user typically owns

These are not autonomous; surface them to the user:
- **Early-stop call**: when to declare the run done before step 311. Currently the user wants to go to step 311 (1 epoch).
- **Host switch**: when to preempt Dedicated tier for Spot (saves $4.70 → $1.95 / h).
- **Tier comparison / new experiment**: M8 launch decision after M5.1 wraps.
- **Reward extension prototype**: the M8.1 / M8.2 implementation in `training_m8_1/` / `training_m8_2/`.

## What an agent should NOT do without asking

- Don't modify `training_m5_1/configs/m5_1_research_paper.yaml` or any training-side code while the run is live.
- Don't push to HF without `pantomiman:$HF_TOK` auth header in the URL (the token's in `~/.cache/huggingface/token`).
- Don't force-push to `experiment_1_h200` (`git push --force`); use regular pushes.
- Don't touch the reward function `training_m5_1/src/rewards/search_r1.py` — that's the M5.1 spec, M8 is a separate `training_m8_*/` overlay.
- Don't run any of the autonomous-loop's wakeup re-arms for cadences past step 311 — once the run finishes the cycle stops.

## Reference: cadence trajectory through C18

| # | Steps | rew_mean | tool_med | len_med | wall (s) | flip rate |
|---|---|---:|---:|---:|---:|---:|
| C4 | 31-40 | 0.171 | 5 | 13.0 K | 376 | — |
| C5 | 41-50 | 0.202 | 3 | 14.6 K | 448 | 37.9 % |
| C6 | 51-60 | 0.224 | 3 | 13.9 K | 412 | 27.9 % |
| C7 | 61-70 | 0.202 | 2 | 13.9 K | 463 | 40.2 % |
| C8 | 71-80 | 0.221 | 3 | 13.9 K | 467 | 33.3 % |
| C9 | 81-90 | 0.228 | 3 | 14.8 K | 484 | 18.6 % (run low) |
| C10 | 91-100 | 0.232 | 3 | 16.1 K | 555 | 26.1 % |
| **C11** | 101-110 | **0.280** (run high) | 3 | 15.8 K | 532 | 42.7 % |
| C12 | 111-120 | 0.247 | 4 | 18.3 K | 606 | 47.4 % |
| C13 | 121-130 | 0.221 | 4 | 20.0 K | 681 | 44.3 % |
| C14 | 131-140 | 0.240 | 5-6 | 23.6 K | 824 | 58.0 % (run high) |
| C15 | 141-150 | 0.242 | 4→3 | 18.0 K | 559 | 40.8 % |
| C16 | 151-160 | 0.256 | 3 | 15.0 K | 411 | 39.6 % |
| C17 | 161-170 | 0.265 | 3→4 | 17.6 K | 551 | 48.6 % |
| C18 | 171-180 | 0.275 | 3-5 wobble | 19.9 K | 660 | 53.4 % |

Single-step run-high: step 105 (C11) = 0.355 and step 170 (C17) = 0.355 (tied).

## Pointers

- Main results doc: [`../report/RESULTS_M5_1_H200.md`](../report/RESULTS_M5_1_H200.md)
- Per-experiment folder convention: [`MILESTONE_5.md`](MILESTONE_5.md) §"Folder layout"
- M8 reward-extension followup: [`../milestone_8/MILESTONE_8.md`](../milestone_8/MILESTONE_8.md)
- Spheron H200 setup runbook: [`../spheron/SETUP_SPHERON.md`](../spheron/SETUP_SPHERON.md)
- F1-reward ceiling diagnosis: [`../report/RESULTS_M5_1_H200.md §9.5`](../report/RESULTS_M5_1_H200.md)

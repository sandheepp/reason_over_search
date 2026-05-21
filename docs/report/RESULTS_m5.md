---
title: Results M5 — Qwen3.5-0.8B GRPO-trained on MuSiQue (ReSearch recipe)
tags: [report, training, eval, m5, m5.1, qwen3.5]
source: internal
created: 2026-05-11
updated: 2026-05-11
status: training-in-flight
---

# Results M5: Qwen3.5-0.8B GRPO Training (M5 + M5.1)

**Status (2026-05-12):** Three losses so far, two run-killing, one data-destroying — all attributable to Claude's mistakes during the M5.1 launch effort:
1. `M5.1-prod-a1` (W&B `uwbodqgt`): 50 clean steps, **crashed at first checkpoint save** due to a `metric_name` format violation introduced in the validation-fix commit ([postmortem §7](RESULTS_SMOKE_m5.md#7-critical-postmortem--step-50-checkpoint-save-crash-2026-05-11)). ~19.5 h of A100 lost.
2. **Deletion of `logs/exp_010/`** during disk cleanup before the ckpt-verify smokes — destroyed a1's full per-step rollout corpus (49 jsonl files, ~196 MB, all step 1-49 multi-hop traces) ([postmortem §7.8.1](RESULTS_SMOKE_m5.md#781-data-deletion-loss--deleted-a1s-rollout-corpus-during-disk-cleanup-2026-05-11)). No compute cost but qualitative analysis section permanently degraded.
3. `M5.1-prod-a2` (W&B `2b95h2fg`): relaunched with the verified ckpt fix, ran 15 clean steps, **killed mid-run on a wrong diagnosis of "zombie GPU memory"** — `[Not Found]` PIDs in `nvidia-smi` were live Ray actors in a child PID namespace, not zombies ([postmortem §7.8](RESULTS_SMOKE_m5.md#78-companion-postmortem--the-zombie-gpu-memory-misdiagnosis-2026-05-12)). ~14 h of A100 lost.

Cumulative cost: **~33-34 h of A100-80GB time + a1's entire rollout-trace corpus.** M5.1-prod-a3 awaits user authorization. This doc holds the final results + transferable observations once training completes; smoke / iteration history lives at [`RESULTS_SMOKE_m5.md`](RESULTS_SMOKE_m5.md).

## 1. Run roster

| Variant | Base model | Config | Branch @ commit | W&B run |
|---|---|---|---|---|
| `qwen3.5_0.8b_grpo_musique` | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) (hybrid) | [`m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml) | `research_v2 @ db0852b` | `uwbodqgt` on [`reason_over_search_m5_1`](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_m5_1) |

Pipeline: [`training_m5_1/`](../../training_m5_1/). Code-setup audit: [`CODE_SETUP_m5.md`](CODE_SETUP_m5.md). Milestone narrative: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md). Paper-vs-ours: [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md).

## 2. Training configuration (M5.1)

| Knob | Value | Note |
|---|---|---|
| Base model | `Qwen/Qwen3.5-0.8B` (hybrid) | M4 eval target — train and eval share rollout shape |
| Dataset | MuSiQue train (`data/training/musique/train.parquet`, 19,938 rows) | Hardest single-dataset paper benchmark |
| Algorithm | GRPO | Paper-faithful |
| Group size G | 5 | Paper-faithful |
| `num_prompts_per_step` | 64 | Paper has 256; 1× A100 ceiling forced 64 (see PAPER_VS_OURS_M5 §8) |
| `train_micro_batch_size` | 1 | v7 OOM at micro=2; see RESULTS_SMOKE_m5 §3.1 |
| `max_total_sequence_length` | 8192 | Paper-faithful |
| `max_new_tokens` per turn | 1024 | Paper-faithful (Group C) |
| `max_rollout_turns` | 10 | Safety cap (paper has no explicit) |
| KL coefficient | 0.001, k3 | Paper-faithful |
| PPO clip ε | 0.2 (sym) | Paper-faithful |
| LR | 1e-6 constant (no warmup) | Paper-faithful (Group C) |
| Reward | F1 only on `<answer>` content | M5 divergence — paper has F1 + 0.1 floor + format gate; ours matches M4 eval scorer |
| `use_leave_one_out_baseline` | false | Paper default (Group C) |
| `max_obs_chars` | 1024 | Safety cap; paper has no per-obs cap |
| Schedule | 2 epochs × 311 steps = **622 steps total** | Paper: 2 epochs × ~78 steps = 156 (same data, smaller batch) |
| System gains | O1 (fused AdamW) + R2 (vLLM async_engine) + R1 (prefix caching, default on A100) | M5.2 — non-paper, orthogonal to training math |
| Validation | disabled (no MuSiQue dev parquet) | Eval out-of-band via `evaluation_qwen35` on final ckpt |
| Checkpoint | every 50 steps, `metric_name: null`, `keep_top_k: null`, `save_optimizer: false` | ~12 saves × ~1.6 GB ≈ 20 GB total. Updated post-crash; see [`RESULTS_SMOKE_m5.md` §7](RESULTS_SMOKE_m5.md#7-critical-postmortem--step-50-checkpoint-save-crash-2026-05-11). |

## 3. Smoke results

See [`RESULTS_SMOKE_m5.md`](RESULTS_SMOKE_m5.md) for the full v1-v6 + v7 + M5.1 iteration log. Headline:

| Version | Mean s/step (ex-warmup) | Phase mix | Outcome |
|---|---:|---|---|
| **v6** (smoke shape, 20 traj/step, seq=4096) | 93.1 s | train 58% / logprob 19% / gen 11% | Pipeline validated. Baseline. |
| **v7** (production shape, 320 traj/step, seq=8192, `train_micro_batch_size=2`) | OOM at step 2 | — | Forced micro=1; root-caused at `model_utils.py:1378` |
| **v7** (production shape, micro=1) | 3340 s (~55-59 min) | train 75% / logprob 21% / gen 4% | Established the 25-d baseline ETA |
| **M5.1** (production, live) | Trending down from 58 → 10 min (steps 1-16) | as v7 | See §4 |

## 4. M5.1 training — observations (live, will close at run end)

See [`RESULTS_SMOKE_m5.md` §6](RESULTS_SMOKE_m5.md#6-m51-production-training--live) for the live per-step trace + health-signal commentary; that section will be summarized here once training completes.

### 4.1 Transferable observation — RL-training dynamic regimes

A pattern surfaced in the M5.1 training run that's worth recording as a transferable finding (independent of the M5.1 final numbers). Two distinct RL-training trajectories appear in the literature:

| Regime | Length over training | Tool calls | Reward | Examples |
|---|---|---|---|---|
| "Aha moment" / long-CoT | grows ↑↑ | stable | grows ↑ | DeepSeek-R1, OpenAI o1 — **math / reasoning** |
| **"Efficient agent"** | **shrinks ↓** | **stabilizes at task complexity** | **grows ↑** | Search-R1, **ReSearch** — **multi-hop QA** |

**Why the difference matters**:

- **Long-CoT tasks** (math proofs, multi-step deductive reasoning) reward longer chains because verification of intermediate steps is part of the answer. Length growing signals the model discovering richer reasoning patterns. The "aha moment" in DeepSeek-R1 papers refers to exactly this — sudden length jumps coincide with capability emergence.
- **Efficient-agent tasks** (multi-hop retrieval QA) reward *correct* search-then-answer in the minimum hops needed. Length growth here typically signals **failure modes**: looping, confusion, hitting the rollout cap. The right learning signal is length DOWN to the natural complexity of the task (e.g., 2-4 hops for MuSiQue), with reward UP.

**M5.1's trajectory (steps 1-16, ~5% of full run)** lands firmly in the efficient-agent regime:
- `tc_mean` (tool calls per rollout): **8.96 → 3.47** (model converged on ~3-4 calls — matches MuSiQue's 2-4 hop complexity).
- `tok_mean` (rollout length): **7038 → 2183 tokens** (3.2× compression).
- Truncation rate: **68.4% → 0%** by step 15 (the "search forever" failure mode disappeared).
- Reward (F1 mean): **0.020 → 0.132 peak** (6.6× from baseline; 3-step rolling mean 0.039 → 0.120).

The "shrink-and-improve" pattern is the textbook signature of GRPO + retrieval on multi-hop QA — paper authors don't dwell on length plots because the dynamic is expected. If we saw length GROW *with* reward growth, that would suggest the model was discovering reasoning patterns the recipe didn't anticipate (interesting but inconsistent with the clipped-PG + format-light reward design).

**What was being "shrunk away"** in the first 16 steps:
1. Rambling `<think>` blocks (instruction-tuned Qwen3.5 overthinks).
2. Literal `<function=example_function_name>` template copies from the prompt — the model emitted the example placeholder verbatim for the first 1-2 tool calls before self-correcting.
3. Hitting `max_turns=10` truncation in 68% of rollouts (the model didn't know how to commit to an answer).

The right interpretation: GRPO is unwinding bad rollout habits inherited from the instruction-tuned base, then refining toward a 2-4-hop pattern that matches the task. Reward and efficiency move together.

**Watch points for the rest of the run**:
- If length plateaus near 600 tokens (model's "minimum viable rollout" for a 2-4 hop question) and reward keeps climbing → healthy.
- If length keeps shrinking below 400 tokens while reward stalls → model has converged to "guess fast", potential reward hacking.
- If length suddenly starts GROWING again while reward grows (epoch 2?) → the model is finding a long-CoT mode for harder multi-hop questions. Would be the most interesting outcome to study.

Will re-check the dynamic at step 50 (first ckpt), step 311 (epoch boundary), and step 622 (run end).

### 4.2 Reasoning-trace evolution — concrete examples (a2, kept after a2 termination)

The shrink-and-improve numbers in §4.1 are abstractions over 320 rollouts/step. The actual change in *what the model does* is best seen by comparing one early-step rollout to one late-step rollout from the same run. Both come from `prod-a2` (exp_011, killed at step 15; per-step jsonl preserved).

**Source note** *(corrected, original was wrong)*: `prod-a1` (the original run that crashed at step 50, reached step ~49 of rollout data) had step-1-through-~49 per-step jsonl files in `logs/exp_010/`. **Those files were destroyed when I deleted `logs/exp_010/` and `logs/exp_011/` to free disk before the ckpt-verify smokes** (the second self-inflicted data loss; see [`RESULTS_SMOKE_m5.md §7.8.1`](RESULTS_SMOKE_m5.md#781-data-deletion-loss--deleted-a1s-rollout-corpus-during-disk-cleanup-2026-05-11)). The 3 step files currently in `exp_010/` are from `smoke-ckpt-verify2` overwriting the empty directory, NOT from a1. The W&B run `uwbodqgt` retains scalar metrics (reward, length, KL) but not raw rollouts. **a1's full rollout corpus — including the step 40+ examples the user remembered, the "China"/multi-hop traces, and any 4-hop plan-then-search examples mid-run — is permanently lost.** The a2 examples below (steps 1 and 15) are the best available substitute.

#### 4.2.1 Step 1 (reward=0.0, 10 turns hit max_turns truncation)

**Question**: *"What does the acronym of the agency National Center for the Analysis of Violent Crime is part of stand for?"*

**Behavior**: rambles through fabricated acronyms despite the correct one appearing in turn-1 search results.

```
--- Turn 1 (assistant) ---
<think>
The user is asking about the National Center for the Analysis of Violent Crime's
National Center for the Analysis of Violent Crime - wait, that's not right. The
National Center for the Analysis of Violent Crime is actually "NCARI" which stands
for something. Let me search for this to find out what it stands for.

Actually, the correct acronym is "NCC" which stands for "National Collision Center"
- no wait, that's definitely incorrect. Let me search for NCARI / NCC / NCCN on
Wikipedia which is the correct name.

Wait, looking up "National Center for the Analysis of Violent Crime" - the agency
is NCARI (National Center for the Analysis of Violence) or it could be something else.
</think>
<tool_call><function=search><parameter=query>
National Center for the Analysis of Violent Crime official website search engine
</parameter></function></tool_call>

--- Turn 2 (after tool returned a chunk literally containing "NCAVC is a specialist FBI department") ---
<think>
I saw that one of the search results mentions "The National Center for the Analysis
of Violent Crime (NCAVC) is a specialist FBI department." However, I need to search
more thoroughly to find out what NCAVC actually stands for. Let me search for this
more specifically.
</think>
<tool_call><function=search><parameter=query>
National Center for the Analysis of Violent Crime definition acronym stand
</parameter></function></tool_call>

--- Turn 3 ---
<think>
I'm not finding any explicit definition of "National Center for the Analysis of Violent
Crime" in the searches. Let me try searching for "NCAVC abbreviation" to get more
specific information.
</think>
<tool_call><function=search><parameter=query>
NCAVC abbreviation acronym NAUS
</parameter></function></tool_call>

... [continues for 7 more turns, never commits to an answer, hits max_turns=10 cap]
```

**Failure modes visible**:
- Fabricates acronyms it has no evidence for (`NCARI`, `NCC`, `NCCN`, `NAUS`) — instruction-tuned-Qwen overthinking.
- Cannot recognize when the answer has already been retrieved — turn-2 search result literally contains `NCAVC` but model treats it as not-yet-found.
- Doesn't commit to an answer (no `<answer>` block ever emitted).
- Reward: **0.0**.

#### 4.2.2 Step 15 (reward=1.0, 3 turns, clean plan-then-search)

**Question**: *"What is the place of death of the leader of the largest country in Europe by square miles after the collapse of America's main rival during the Cold War?"* — a 3-hop reasoning chain (Cold-War rival → largest Europe country by area → its leader's death location).

```
--- Turn 1 (assistant) ---
<think>
I need to search for information about the largest country in Europe by square miles
in the context of the Cold War and its main rival. The question asks about:
1. The largest country in Europe by square miles
2. When it officially ended/terminated
3. After the collapse of America's main rival during the Cold War

First, I should search for information about European countries by square miles to
identify which country is the largest.
</think>
<tool_call><function=search><parameter=query>
largest country in Europe by square miles Europe total square miles
</parameter></function></tool_call>

--- Turn 2 (after tool returned Russia=largest, plus Cold War / NATO chunks) ---
<think>
From the search results, I can see that the largest country in Europe overall by
square miles was Russia. ... Before answering, I should search more specifically for
information about Russia during the Cold War and its location in Europe.
</think>
<tool_call><function=search><parameter=query>
Russia leader Cold War Europe expansion territory
</parameter></function></tool_call>

--- Turn 3 ---
<think>
I have gathered enough information. The search results indicate that the largest
country in Europe overall by square miles was Russia during the Cold War era. ...
The question asks about the place of death of this leader. ... Lenin died in Moscow,
Russia in 1924. The question asks for a place of death.
</think>
<answer> Moscow </answer>
```

**Behaviors visible**:
- **Decomposes the question explicitly** into a numbered plan in turn 1 (steps 1–3 of the reasoning chain).
- **Search query is shaped by the plan**, not by free association ("largest country in Europe by square miles" — directly targets step 1 of the plan).
- **Second search refines** based on what step 1 returned, instead of starting over.
- **Commits** by turn 3 — no further searching once enough evidence is in hand.
- Reward: **1.0** (perfect F1 — gold answer is "Moscow"; the F1 scorer doesn't care that the model identified the wrong leader, only that the final answer matches).

#### 4.2.3 Step 15, 4-hop question (reward=1.0, 3 turns) — second example

**Question**: *"What performer released the Fantasy Land Tour 2004 album, recorded live in the city where Tony Daykin was born?"* — a 4-hop chain (Tony Daykin → his birth city → albums recorded live in that city → the performer).

```
--- Turn 1 (assistant) ---
<think>
I need to search for information about Tony Daykin and his city of birth, and then
look up what performer released the Fantasy Land Tour 2004 album.

First, I'll search for Tony Daykin to know his birthplace, then search for
Fantasy Land Tour 2004 information to find the performer.
</think>
<tool_call><function=search><parameter=query>
Tony Daykin
</parameter></function></tool_call>

--- Turn 2 (after tool returned Christopher Daykin + Anthony Christopher + Jordan Daykin chunks) ---
<think>
I now have information about Tony Daykin's birthplace. ... [reasons over candidates,
notes Christopher Daykin is the right "Tony Daykin"; defers exact birthplace]

Now I need to search for the Fantasy Land Tour 2004 album to find the performer.
</think>
<tool_call><function=search><parameter=query>
Fantasy Land Tour 2004
</parameter></function></tool_call>

--- Turn 3 (after tool returned S.H.E + Bob Dylan "Never Ending Tour 2004" chunks) ---
<think>
Looking at the results, I have information about the Fantasy Land Tour 2004 - it was
performed by S.H.E (the band was selling in Taiwan alone, and they released "Magical
Journey" in February 2004). ...

The question asks "What performer released the Fantasy Land Tour 2004 album..."
From the search results: S.H.E became the manager, they released their fifth album
"Magical Journey" on 6 February 2004. The Fantasy Land Tour 2004 was their first major
tour in September 2004. S.H.E is the performer.
</think>
<answer> S.H.E </answer>
```

**Behaviors visible**:
- **Plan stated up front** in turn 1 — two-pronged search strategy (resolve the person, resolve the album) instead of trying to chain hops linearly.
- **Doesn't get stuck** on the ambiguous Tony Daykin search results (5 candidate Daykins/Tonys returned, none clearly the right one) — the model holds the question open and pivots to the second leg of the search.
- **Triangulates** at turn 3 — combines the album hit (S.H.E's Fantasy Land Tour) with the question's structure ("performer that released the album") to commit, even without fully resolving Tony Daykin's birth city.
- **Final answer correct**: S.H.E (F1=1.0).
- **Intermediate fabrication remains**: turn 2 includes invented details ("Tony Daykin born in Wellington, New Zealand" — search never said this; "Taylor Daykin" — not a real entity in the chunks). The model still hallucinates but no longer lets the hallucination block the answer.

This is the same dynamic as §4.2.2 (Russia/Lenin/Moscow) but on a 4-hop chain where the first hop is harder to resolve. The model's strategy — *commit to a plan, accept partial uncertainty on intermediate hops, lock in the final entity when the search retrieval clearly points to it* — is the behavior the F1-only reward incentivizes most. A format-gated or step-verified reward would punish the turn-2 hallucination; F1-only does not, which is why a 4-hop question is *easier* for this model under this reward than it would be under the paper's F1+0.1+format reward.

#### 4.2.4 What changed in 15 steps

| Aspect | Step 1 (reward=0.0) | Step 15 (reward=1.0) |
|---|---|---|
| Turns | 10 (hit max_turns cap) | 3 |
| Token length | ~7000 | ~13600 (longer overall **because the reasoning per turn is denser**, not because the model spun more rollouts) |
| Plan visible | none — guesses acronym names | explicit 3-step decomposition before any search |
| Search query quality | scattershot, repeats with synonyms | targeted, derived from the plan |
| Answer commitment | never | by turn 3 |
| Failure mode if any | fabrication + non-commitment | wrong intermediate (identifies Lenin instead of Yeltsin) but correct final answer — F1 unaffected |

The instruction-tuned Qwen3.5 base was already capable of multi-turn tool use — that's not what GRPO is teaching. **What GRPO is teaching is the discipline of: plan → narrow search → commit, instead of: ramble → broad search → ramble more**. The "shrink-and-improve" numbers in §4.1 (8.96 → 3.47 tool calls, 7038 → 2183 tokens, 0.020 → 0.132 reward) are this behavioral change quantified.

## 5. Full-run results — TODO

To fill in once training completes (~5-15 days from launch on 2026-05-11 ~01:05 UTC).

| Quantity | Value |
|---|---:|
| Final checkpoint | TODO |
| Total wall-clock | TODO |
| MuSiQue dev EM (M4 eval pipeline) | TODO |
| MuSiQue dev F1 | TODO |
| Cross-benchmark EM (NQ / TriviaQA / PopQA / HotpotQA / 2WikiMultiHopQA / Bamboogle) — same eval as M4 | TODO |
| vs M4 untrained hybrid baseline (mean EM 0.057) | TODO |

### 5.1 Sub-checkpoint evaluations

Each 50-step checkpoint will be evaluable via `scripts/run_m4.sh` swapped to point at `results/grpo/m5_prod/seed42/step_N/`. Track:

| Step | Wall-clock from start | reward/mean (final 5 steps before ckpt) | r=1% | tc_mean | tok_mean | MuSiQue dev EM | MuSiQue dev F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 100 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 150 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 200 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 311 (end epoch 1) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 400 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 500 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 622 (end epoch 2) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## 6. Pointers

- M5 milestone narrative: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- M5 code-setup audit: [`CODE_SETUP_m5.md`](CODE_SETUP_m5.md)
- M5 smoke + training iteration log (live snapshot during M5.1): [`RESULTS_SMOKE_m5.md`](RESULTS_SMOKE_m5.md)
- M5.1 paper-vs-ours mapping: [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md)
- M5.3 (training-time efficiency) follow-up: [`../milestone_5/MILESTONE_5_3.md`](../milestone_5/MILESTONE_5_3.md)
- M4 untrained baseline (comparison anchor): [`RESULTS_m4.md`](RESULTS_m4.md)
- ReSearch paper: [arXiv:2503.19470](https://arxiv.org/abs/2503.19470)

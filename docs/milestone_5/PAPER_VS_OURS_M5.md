---
title: PAPER vs OURS — M5 / M5.1 (ReSearch paper recipe on Qwen3.5-0.8B / MuSiQue / F1-only)
tags: [milestone, training, qwen3.5, m5, m5.1, paper-vs-ours, research-paper, mapping]
source: internal
created: 2026-05-10
updated: 2026-05-10
---

# Paper-vs-ours mapping: M5 / M5.1

Clause-by-clause mapping from the [ReSearch paper](https://arxiv.org/abs/2503.19470) and the [`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch) reference configs to our M5.1 NeMo-RL yaml at [`training_m5_1/configs/m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml) (to be authored in M5.1 step 6). Status: **spine only**; sections marked **TODO** populate as the paper-read + codebase-read work lands. Companion to [`CODE_SETUP_m5.md`](../report/CODE_SETUP_m5.md) and [`MILESTONE_5.md`](MILESTONE_5.md).

The two intentional divergences (carried from M4 + Phase-1 finding #4) are flagged in **RED** below.

## Sources of truth

1. ReSearch paper, [arXiv:2503.19470](https://arxiv.org/abs/2503.19470); notes at [`../papers/2503.19470_research.md`](../papers/2503.19470_research.md) (existing).
2. ReSearch reference code, [`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch) — concrete config values, retriever HTTP contract, prompt strings, rollout shape.
3. Phase-1 [`research`](https://github.com/pantomiman/research) repo Qwen3-0.6B verl runs — transferable retriever / rollout / GRPO knob choices already validated against this codebase.
4. M2 paper-to-NeMo-RL mapping work: [`../training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md), [`../training/VERL_REFERENCE.md`](../training/VERL_REFERENCE.md), [`../training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md).

## 1. Model and data

| Concern | Paper | Ours (M5.1) | Note |
|---|---|---|---|
| Backbone | Qwen2.5-7B / Qwen2.5-32B-Instruct / Llama3.1-8B variants | **Qwen3.5-0.8B** (hybrid) | Smallest viable: ~10x cheaper training; same Qwen3.5 family as M4 eval baseline |
| Variant | both base and instruct | **hybrid only** (initial; base as ablation later) | Hybrid has the better untrained EM floor on M4 (mean 0.057 / dataset, n=100, M4.2 minimal) |
| Training data | HotpotQA + 2Wiki mix (paper §B) | **MuSiQue only** | Single-dataset; hardest of the four paper benchmarks (M1 baseline EM 0.124 → largest headroom) |
| Eval data | NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle | identical 7-benchmark suite | Eval continues via M4 pipeline; M5 produces a checkpoint, eval reads it as-is |

## 2. Reward and answer format (the two divergences)

| Concern | Paper | Ours (M5.1) | Divergence? |
|---|---|---|---|
| Reward function | **F1 on `<answer>X</answer>`** content; **0.1 partial-credit floor** when format valid but F1=0 (paper §2.3 Eq. 2; ref code [`re_search.py:compute_score`](https://raw.githubusercontent.com/Agent-RL/ReCall/re-search/src/verl/utils/reward_score/re_search.py)) | **F1-only on `<answer>...</answer>` content**; **no 0.1 floor** | **YES (M5.1 divergence #1, narrowed: we drop the 0.1 valid-format floor; still F1 on the answer)** |
| Format reward coefficient | 0.1 floor for valid-format-but-wrong-answer | 0 (dropped per Phase-1 finding #4) | YES |
| Answer wrap | `<answer>...\boxed{X}...</answer>` (paper §2.2) | **plain `<answer>X</answer>`** | **YES (M5.1 divergence #2)** |
| Answer-extract source | last `<answer>` block (ref code) | **first** `<answer>` block (M4 convention, mirror of [`evaluation_qwen35/flashrag/search_r1/reward.py:extract_solution`](../../evaluation_qwen35/flashrag/search_r1/reward.py)) | YES (M4 carry; M4 `extract_solution` falls back to the last match via the regex, so this aligns) |
| Token normalisation | SQuAD-style (lowercase + strip punct + strip articles + collapse whitespace) | identical (re-export from M4 `normalize_answer`) | no |

**Correction (2026-05-10)**: earlier drafts of this doc described the paper reward as "EM + partial credit". The ref-code inspection (`Agent-RL/ReCall@re-search` branch, [`src/verl/utils/reward_score/re_search.py`](https://raw.githubusercontent.com/Agent-RL/ReCall/re-search/src/verl/utils/reward_score/re_search.py) `compute_score`) shows the paper actually uses **F1 plus a 0.1 valid-format floor**. So our M5.1 reward is **closer to the paper than originally thought** — same scorer (F1), we just drop the floor.

**Why no 0.1 floor (Phase-1 finding #4)**: the floor masks the tool-use learning signal (3 to 6 pp gap between tool-using and no-tool runs in Phase-1). Dropping it preserves the discriminative F1 signal at 0.8B.

**Why no `\boxed{}`**: M4 already uses plain `<answer>X</answer>` in the eval prompts; M5.1 keeps that for train/eval parity. The M4 `extract_solution` accepts both shapes, so eval would handle either; we just pick the simpler one for the rollout prompt.

## 3. GRPO algorithm hyperparameters

Filled from the [ReSearch paper](https://arxiv.org/abs/2503.19470) §B + Table 4 and [`Agent-RL/ReCall@re-search`](https://github.com/Agent-RL/ReCall/tree/re-search) reference configs ([`scripts/train/train.sh`](https://raw.githubusercontent.com/Agent-RL/ReCall/re-search/scripts/train/train.sh), [`src/verl/trainer/config/ppo_trainer.yaml`](https://raw.githubusercontent.com/Agent-RL/ReCall/re-search/src/verl/trainer/config/ppo_trainer.yaml)).

| Knob | Paper | Ref code | Ours (M5.1 yaml key) | Status |
|---|---|---|---|---|
| Algorithm | GRPO | `adv_estimator=grpo` | `grpo.adv_estimator.name=grpo` | match |
| Group size G | 5 (Table 4) | `ROLLOUT_N=5` ([train.sh L11/L84](https://raw.githubusercontent.com/Agent-RL/ReCall/re-search/scripts/train/train.sh)) | `grpo.num_generations_per_prompt: 5` | match |
| Prompts/step | 256 (Table 4) | `TRAIN_BATCH_SIZE=256` (train.sh L5) | `grpo.num_prompts_per_step: 256` | TBD; ours capped by 1× A100 batch shape — likely **64 or 128 per step** to fit memory, document the divergence |
| Trajectories/step | 256 × 5 = 1280 | 1280 | `policy.train_global_batch_size` | derived |
| KL type | low-variance KL ≡ k3 | `kl_loss_type=low_var_kl` (train.sh L80) | `loss_fn.reference_policy_kl_type: k3` | match (NeMo-RL k3 ≡ verl `low_var_kl`) |
| KL coefficient β | 0.001 (Table 4) | `kl_loss_coef=0.001` + `kl_ctrl.kl_coef=0.001` (train.sh L66, L79) | `loss_fn.reference_policy_kl_penalty: 0.001` | match |
| PPO clip ε | 0.2 (Table 4) | `clip_ratio=0.2` (verl default) | `loss_fn.ratio_clip_min: 0.2`, `ratio_clip_max: 0.2` | match (symmetric ±0.2) |
| Advantage normalisation | not stated | verl GRPO default: group-mean + std normalisation | `grpo.normalize_rewards: true` | match |
| Baseline | group mean | verl GRPO default (not LOO) | `grpo.use_leave_one_out_baseline: true` (ours) | **mild divergence: NeMo-RL prefers LOO baseline**; same group-relative idea, slightly lower variance |
| Learning rate | 1e-6 (Table 4) | `optim.lr=1e-6` (train.sh L75) | `policy.optimizer.kwargs.lr: 1.0e-6` | match |
| Warmup | not stated | `lr_warmup_steps_ratio=0.0` (verl default) | **drop warmup for production** (smoke kept warmup=14 for the curve) | NEW divergence to fix in m5_1_research_paper.yaml |
| LR schedule | not stated | `warmup_style=constant` (verl default) | `ConstantLR` | match |
| Weight decay | not stated | AdamW default (verl FSDP) | 0.01 (M2 carry) | minor divergence; document |
| Optimizer | not stated | AdamW (verl FSDP default) | `torch.optim.AdamW` | match |
| Total steps | 2 epochs (paper §3.1) on MuSiQue 19,938 → ~156 steps at bs=256 | `TOTAL_EPOCHS=2` (train.sh L19) | `grpo.max_num_steps: ~156` (if we hit bs=256); higher if our bs is smaller | epoch-driven; tune at smoke landing |

### Open question on prompts/step

Paper batch = 256 prompts. Our smoke is at 4 (smoke shape). For production on 1× A100-80GB, the practical ceiling is rollout latency: 256 prompts × 5 gen × ~30 s/rollout would dominate the step. Likely M5.1 production lands at 32-128 prompts/step depending on smoke wall-clock. Decision after smoke result.

## 4. Generation / rollout shape

| Concern | Paper / ref code | M4 eval (locked) | Ours (M5.1 training) | Note |
|---|---|---|---|---|
| `max_response_length` (per-rollout cap on TOTAL generated tokens across all turns) | **8192** (`MAX_RESPONSE_LENGTH=8192` in train.sh L7) | n/a (eval is multi-turn; each turn uncapped at the per-generation level, full 4096 window per turn) | smoke: 500 per-turn (M2 anchor); production: needs decision | **architectural mismatch**: paper's 8192 is total rollout budget across turns; NeMo-RL's `max_new_tokens` is per-generation-call. See note below. |
| `max_prompt_length` (the initial prompt, before any tool responses) | 512 (`MAX_PROMPT_LENGTH=512` in train.sh L6) | n/a | 4096 (`policy.max_total_sequence_length`, covers full context window including tool responses) | **architectural mismatch** (paper) — paper's 512 is initial prompt only; NeMo-RL's `max_total_sequence_length` is the whole transformer context window |
| `max_search_turns` / `max_rollout_turns` | **no explicit cap**; implicit bound from 8192-token response budget | 5 | 5 (NeMo-RL env requires an explicit cap) | minor divergence; ours = M4 carry. 5 turns × ~500 tokens ≈ 2500 tokens ≪ paper's 8192 implicit budget so we end up tighter |
| `retrieval_topk` | 5 (hardcoded in [vllm_rollout.py L261](https://raw.githubusercontent.com/Agent-RL/ReCall/re-search/src/verl/workers/rollout/vllm_rollout/vllm_rollout.py) — not a YAML knob) | 5 | 5 | match |
| `max_obs_length` (per-chunk or total cap on tool-response text) | **no truncation** — raw retrieval text appended | 256 tokens per chunk (M4 v3 lock) | 480 chars / chunk (= ~120 tokens) | divergence; ours is stricter to fit a wider prompt budget. Could relax to no-cap if 480 is too aggressive |
| Rollout temperature | 1.0 (Table 4; verl default not overridden) | n/a (eval is greedy) | 1.0 | match |
| Rollout top_p | 1.0 (verl default) | n/a | 1.0 | match |
| Rollout top_k | -1 (disabled; verl default) | n/a | null | match |
| Stop strings (first turn fallback) | n/a | `</tool_call>` / `</answer>` / `<\|im_end\|>` / `<\|endoftext\|>` | identical | env's `next_stop_strings` covers subsequent turns |
| `enable_thinking` | n/a (paper predates Qwen3.5) | True | True | M4 lock; opens `<think>\n` generation prefix |

### Note on the per-turn vs total response budget

Paper's `max_response_length=8192` is the **total** response budget summed across all rollout turns in verl's `vLLMRolloutWithSearch` loop ([`vllm_rollout.py:253+`](https://raw.githubusercontent.com/Agent-RL/ReCall/re-search/src/verl/workers/rollout/vllm_rollout/vllm_rollout.py)). The rollout terminates when either no `</search>` is emitted (terminal) or cumulative tokens hit 8192.

NeMo-RL's env loop drives each turn as a separate `policy.generation.max_new_tokens` call. So our analog of paper's 8192 is `max_new_tokens × max_search_turns`. For paper-faithful budget at 5 turns: `max_new_tokens ≥ 8192/5 ≈ 1640`. Our smoke at 500 leaves headroom for short rollouts but truncates long ones. Production needs at least 1024 to be paper-fair; 1640 is the closest match.

This is the cleanest single-knob change between smoke and production yaml.

## 5. Observation formatting (the tool-response wrap)

Hard alignment requirement: train-time and eval-time tool-response strings must be byte-identical, else the rollout distribution drifts between train and eval ([`CODE_SETUP_m3.md`](../report/CODE_SETUP_m3.md) §3 documents the M3 14-fix audit that fixed this for Qwen3-0.6B). The M5.1 env uses the same `format_docs_qwen_native` helper as eval; verified by [`training_m5_1/tests/test_format_helpers.py`](../../training_m5_1/tests/test_format_helpers.py).

```text
<|im_end|>
<|im_start|>user
<tool_response>
{doc_1 body, capped at 480 chars per chunk}
</tool_response>
<tool_response>
{doc_2 body, capped at 480 chars per chunk}
</tool_response>
...
<|im_end|>
<|im_start|>assistant
```

Per-chunk cap is 480 chars (≈120 tokens at 4 chars/token), matching the M4.1 v3 lock. Total of N `<tool_response>` blocks per turn, where N = `retrieval_topk` = 5.

## 6. Prompt template

Carried from M4 dynamically: [`training_m5_1/scripts/sync_m4_prompts.py`](../../training_m5_1/scripts/sync_m4_prompts.py) materialises whichever `QWEN35_TEMPLATES[mode]` M4.4 locks.

**M4.4 Phase 1b winner (locked 2026-05-10): `qwen35_terse`** (Δ +0.0436 over `qwen35_minimal` baseline, mean EM 0.103 across 7 datasets at n=300 — closes the M3 cross-family gap). M5.1 smoke and production both reference this template. Content (after `{prompt}` → `{}` translation for our positional `.format(question)`):

```
Use the `search` tool to look up facts as needed. When you have the answer, write it inside <answer> and </answer>. For example, <answer> Beijing </answer>.
Question: {}
```

170 chars, user-locus, no system message, `tools=[QWEN35_SEARCH_TOOL]` auto-injected by the chat template. Roughly 1/3 the length of the next-best mode (`qwen35_minimal` at 573 chars). The terse win pattern reduces pre-question scaffolding, which the M4.2 audit traced as the dominant prompt-bloat failure mode for 0.8B.

To re-sync if a future M4.x finds a better mode:

```bash
python training_m5_1/scripts/sync_m4_prompts.py --mode <new-winner-key>
```

(The `qwen35_minimal_no_system` mode additionally needs a processor-level `pass_tools=False` knob that does not yet exist; the script warns. Not relevant for the current `qwen35_terse` winner.)

## 7. Schedule and wall-clock projection

Partial fill from the paper-mapping work; per-step wall-clock comes from the M5.1 smoke (in flight at the time of writing). Paper-faithful production shape would be:

| Quantity | Value | Source |
|---|---:|---|
| Paper schedule (steps) | ~156 (2 epochs × 19,938 MuSiQue rows / 256 prompts/step) | paper §3.1 + train.sh `TOTAL_EPOCHS=2` |
| Trajectories/step (paper) | 256 prompts × 5 group = **1,280** | paper Table 4 |
| Trajectories/step (ours likely, after smoke) | ~640 (128 prompts × 5) or 320 (64 × 5), driven by 1×A100 batch-shape ceiling | smoke smoke result |
| Per-step wall-clock target | ≤25 s/step at 20 traj smoke; production should land at ≤90 s/step at 320 traj | M2 anchor: 57 s/step at 20 traj on Qwen3.5-2B; scaling: ÷2.5 for 0.8B vs 2B, ×16 for traj count, ≈90 s |
| Total wall-clock (production) | ~156 × ~90 s = ~4 h | derived |
| Cost @ \$1.20/h on 1× A100-80GB | ~\$5 | derived |
| Fits ≤10 h target? | **likely Y** with margin | [`../todo/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md) §thesis-affordable-budget |

If over budget: cheapest knobs to cut, in order of expected wins, are `max_response_length` (cubic-ish on rollout time) and `num_prompts_per_step` (linear on rollout time, but degrades GRPO group-baseline if dropped below ~32).

**Big caveat**: paper's 8192-token total response budget vs ours of `max_new_tokens × max_search_turns ≈ 2500` is a ~3× tighter rollout. If rollouts truncate often, raise `max_new_tokens` to 1024 or 1640 for paper-fairness (see §4 note).

## 8. Resolved decisions (2026-05-10 evening)

After the M5 smoke step-1 success + the comprehensive paper-mapping audit, the following Group-C calls are locked. All target the m5_1_research_paper.yaml; a subset that's safe at smoke shape is also applied to m5_smoke.yaml ahead of the M5.1 production smoke.

| Knob | Locked value | Reason |
|---|---|---|
| `use_leave_one_out_baseline` | **false** | Paper uses group-mean baseline (verl GRPO default); LOO was a non-paper carry from M2 |
| `lr_warmup_steps_ratio` | **0.0** | Paper uses verl default (no warmup); 0.285-ratio warmup was the M2 carry, dropped |
| `policy.generation.max_new_tokens` | **1024** | Per-turn cap; total rollout budget = 1024 × ~5 turns ≈ 5000 tokens, in the ballpark of paper's 8192 implicit cap |
| `env.search_r1.max_obs_chars` | **1024** (smoke) / consider unbounded for production | Paper has no per-obs cap; smoke at 1024 is the safety net, may relax in production yaml |
| `env.search_r1.max_turns` | **10** | Paper has no explicit cap; bounded by `max_total_sequence_length` |
| `policy.max_total_sequence_length` | smoke: **4096** (memory-safe); production: **8192** | Paper effectively uses ~8704 total context (512 prompt + 8192 response) |
| `policy.train_micro_batch_size` | smoke: **2**; production: best-fit | Step-2 OOM at micro=4 by 0.31 GB; production yaml tunes to memory ceiling |
| `policy.generation.vllm_cfg.gpu_memory_utilization` | smoke: **0.5**; production: best-fit | Lower than M2's 0.7 to leave more headroom for the policy worker between wake/sleep cycles |
| **Reward formula** | F1-only on `<answer>X</answer>` content (no 0.1 floor) | Phase-1 finding #4 (the floor masks tool-use signal at 0.8B) |
| **Answer wrap** | plain `<answer>X</answer>` (no `\boxed{}`) | M4 lock; eval scorer accepts both |
| **Action tags** | Qwen3.5-native `<tool_call>` / `<tool_response>` | In-distribution Qwen3.5 post-training |
| **Training data** | MuSiQue train (19,938 rows) | Confirmed matches paper §3.1 + `Agent-RL/ReCall@re-search/data/prepare_musique.py` |

## 9. Deferred system gains (after M5.1 production yaml lands)

These do not change training mathematics; pure throughput. Probe these AFTER the production yaml is correct and the production-shape smoke gives a baseline s/step number.

| Lever | Where | Expected gain | Risk |
|---|---|---|---|
| `torch.compile` on policy fwd/bwd | NeMo-RL `policy.dtensor_cfg` (check upstream knob) | 10-30% step time (1.x training-side) | Compilation cache must persist across runs; Qwen3.5 hybrid layers may not compile cleanly |
| vLLM AOT compile cache reuse | already on (`/root/.cache/vllm_0/torch_compile_cache/`) | already capturing; just verify cache survives Vast instance restarts | Cache is per-(model, vLLM version, GPU arch); invalidates on any of those changing |
| `async_grpo.enabled: true` | NeMo-RL grpo config | overlap rollout-generation with policy-train step | Slightly off-policy advantage (bounded by `max_trajectory_age_steps`); adds complexity |
| `sequence_packing.enabled: true` | NeMo-RL policy config | 20-40% on training step (per upstream docs) | **BLOCKED**: Qwen3.5 GatedDeltaNet kernel crashes with packing ([training/fix/CHANGES.md §5](../../training/fix/CHANGES.md)); upstream NeMo-RL bug |
| Higher `vllm_cfg.gpu_memory_utilization` | YAML | bigger KV cache → bigger rollout concurrency | OOM during wake/sleep transitions (we already learned 0.7 was unsafe at micro=4) |
| `policy.generation.colocated.enabled: true` (already on) | YAML | already saves a process boundary | n/a — verified on |
| Flash Attention 3 | vendor venv pin | 5-15% on attention layers | Requires nemo_rl venv rebuild + AOT cache invalidate |

## 10. Pointers

- M5 milestone narrative: [`MILESTONE_5.md`](MILESTONE_5.md)
- M5 code setup: [`../report/CODE_SETUP_m5.md`](../report/CODE_SETUP_m5.md)
- M5 smoke results: [`../report/RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md)
- M2 paper-to-NeMo mapping (foundation): [`../training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md), [`../training/VERL_REFERENCE.md`](../training/VERL_REFERENCE.md), [`../training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md)
- M4 eval pipeline (the rollout-shape source of truth): [`../milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md), [`../report/CODE_SETUP_m4.md`](../report/CODE_SETUP_m4.md)
- ReSearch paper: [arXiv:2503.19470](https://arxiv.org/abs/2503.19470), notes at [`../papers/2503.19470_research.md`](../papers/2503.19470_research.md)
- ReSearch official codebase: [`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch)

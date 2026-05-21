---
title: PAPER VS OURS TRAINING
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-06
---

# Paper vs Ours — Training

Training-side counterpart to [`docs/eval/PAPER_VS_OURS_AUDIT.md`](../eval/PAPER_VS_OURS_AUDIT.md). Documents every knowing departure from Search-R1's training setup, plus the rationale.

Sources for paper numbers: [Search-R1 arXiv 2503.09516](https://arxiv.org/abs/2503.09516) (Appendix B.2), [Search-R1 GitHub](https://github.com/PeterGriffinJin/Search-R1).

---

## 1. Model

| | Paper | Ours |
|---|---|---|
| Family | Qwen2.5 | **Qwen3.5** |
| Size | 3B (also 7B) | **2B** |
| Variants | base, instruct | **base, hybrid** (Qwen3.5-2B + soft-switch reasoning, no separate Instruct) |
| Pre-training tool format | none consistent | `<tool_call>` / `<tool_response>` (Hermes-style) |
| Native reasoning tag | none | `<think>` (with `enable_thinking` toggle) |

**Why we differ:** Qwen3.5 is the current state-of-the-art for the Qwen family at this scale and supports tool use natively; verl does not support Qwen3.5, which is why we port to NeMo-RL. The 2B size keeps a single A100 80GB sufficient for 1-GPU training; the smaller param count is offset by Qwen3.5's stronger pre-training.

## 2. Framework

| | Paper | Ours |
|---|---|---|
| RL library | [`verl`](https://github.com/volcengine/verl) | [`NeMo-RL`](https://github.com/NVIDIA-NeMo/RL) |
| Rollout backend | vLLM | vLLM |
| Distributed backend | FSDP + CPU offload | DTensor (default) — Megatron available as alternative |

**Why we differ:** verl does not have Qwen3.5 support as of May 2026. NeMo-RL supports GRPO + RLVR for Qwen3.5 out of the box ([NeMo-RL releases](https://github.com/NVIDIA-NeMo/RL/releases)).

## 3. Reward function

**Paper:** rule-based outcome-only reward, exact-match against gold answer:

$$r_\phi(x, y) = \mathrm{EM}(a_\text{pred}, a_\text{gold})$$

No format penalty, no process reward, no learned reward model. ([arXiv 2503.09516, §3.4](https://arxiv.org/html/2503.09516)). The paper is explicit about excluding shaping:

> "Unlike Guo et al. (2025), we do not incorporate format rewards, as our learned model already demonstrates strong structural adherence."

**Ours:** matches the paper — **pure EM, no shaping**. `r = 1.0` if normalized-EM hits, else `r = 0.0`.

> **Paper-vs-Search-R1-repo gap (important).** The Search-R1 GitHub repo ships **two** reward functions: [`qa_em.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py) (paper-faithful, EM only) and [`qa_em_format.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em_format.py) (a 6-tier shaped variant exposing `structure_format_score`, `final_format_score`, `retrieval_score`). The shaped variant is *not* what the paper describes. Earlier in this project we ported `qa_em_format.py` (with non-zero shaping defaults of 0.2 / 0.1 / 0.1, the values exposed in FlashRAG's CLI flags) and described it as "identical to the paper" — that was a documentation error caught in early-May smoke testing (see [`docs/training/SMOKE_RESULTS_2026-05-06.md`](SMOKE_RESULTS_2026-05-06.md) for the trail; raw observations are in the archived 2026-05-02 + 2026-05-04 smoke runs at [`docs/archive/training/`](../archive/training/)). The shaping was producing visible partial-credit reward signal on rollouts where EM=0, masking what was actually a flat learning curve.
>
> **Resolution:** the multi-tier scaffold remains in [`training/src/rewards/search_r1.py`](../../training/src/rewards/search_r1.py) (and the eval-side mirror), but **all three shaping coefficients default to 0.0**, collapsing the function to pure EM. The scaffold stays so that M3 ablations can re-introduce shaping by passing non-zero coefficients explicitly without rewriting the state-machine format walker.

The EM scorer is **byte-identical** to [`evaluation_search_r1/flashrag/search_r1/reward.py`](../../evaluation_search_r1/flashrag/search_r1/reward.py) (the SQuAD-canonical normaliser used in Milestone 1 eval): port at [`training/src/rewards/search_r1.py`](../../training/src/rewards/search_r1.py). Verified by [`training/tests/test_reward_parity.py`](../../training/tests/test_reward_parity.py). Training-time reward and post-training EM are computed by the same code.

## 4. Chat template

| | Paper | Ours |
|---|---|---|
| Retrieval call | `<search>query</search>` | **`<tool_call><function=search><parameter=query>...</parameter></function></tool_call>`** (Qwen3.5 XML-nested) |
| Retrieved doc | `<information>...</information>` | **`<tool_response>...</tool_response>`** in a synthetic user turn |
| Reasoning | `<think>...</think>` | `<think>...</think>` (identical) |
| Final answer | `<answer>...</answer>` | `<answer>...</answer>` (identical — required by EM scorer) |

Full rationale + verbatim jinja in [`CHAT_TEMPLATE.md`](CHAT_TEMPLATE.md).

## 5. Validation set & cadence

> ⚠️ **Currently disabled (first-pass training)**: both YAMLs have `grpo.val_period: 0` and `data.validation: null`; checkpointing is also off. Mechanics-verification first; re-enable per [`VALIDATION.md §7`](VALIDATION.md#7-re-enabling-validation-planned-not-active).

See [`VALIDATION.md`](VALIDATION.md) for the full plan. Summary (the values once re-enabled):

| | Paper / verl yaml | Ours |
|---|---|---|
| Val datasets | NQ + HotpotQA (held-out portions) — used as in-loop validation | Match paper |
| Save cadence | every 100 steps (`trainer.save_freq=100`) | match (`checkpointing.save_period: 100`) |
| **Test cadence** | **every 100 steps (`trainer.test_freq=100`)** — confirmed in [`Search-R1/scripts/nq_hotpotqa/v0.2/train_grpo.sh:69`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/nq_hotpotqa/v0.2/train_grpo.sh) | match (validation co-runs with each checkpoint save) |
| **Total steps** | **1005** (`trainer.total_training_steps=1005` in v0.2 verl yaml — supersedes the paper text's "500" claim, which appears to be an Appendix B.2 summary inconsistent with the published-checkpoint training run) | match (`grpo.max_num_steps: 1005`) |
| Val-before-train | `+trainer.val_before_train=true` (run val at step 0 as baseline) | match — log a step-0 val point for sanity |
| Logged metrics | EM, reward, format-validity | EM, reward, format-validity, length-truncation, mean answer tokens |

## 6. Hyperparameters

Paper values from Appendix B.2; verl values from [`Search-R1/scripts/nq_hotpotqa/v0.2/train_grpo.sh`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/nq_hotpotqa/v0.2/train_grpo.sh) (the EM-only baseline that produced the published GRPO checkpoints).

| Hyperparameter | Paper / verl | Ours | Notes |
|---|---|---|---|
| Learning rate | `1e-6` | `1e-6` | match |
| Warmup ratio | `0.285` | `0.285` (286 LinearLR iters of 1005) | match (unusually high; revisit after first run) |
| **Total training steps** | **`1005`** (verl `total_training_steps=1005`; supersedes paper text's "500") | **`1005`** | match — see §5 |
| Trajectories per step | **2560** (verl: `train_batch_size=512` prompts × `n_agent=5`) | **510** (102 prompts × 5 generations) | **5× fewer per step**. AND verl mini-batches into 10 updates/step vs our 1; net **~10× fewer gradient updates over 1005 steps**. Largest unmodelled gap; see [`docs/edu/BATCH_MATH.md`](../edu/BATCH_MATH.md). First-pass accepts the divergence. |
| Gradient updates per step | **10** (verl `ppo_mini_batch_size=256` over 2560 → 10 PPO mini-batches) | **1** (gbs=510 == prompts × gen → one optimizer.step()) | The mechanism behind the row above. PPO clip bounds per-step parameter movement, but verl gets ~10× the optimization work per training step. |
| PPO mini/micro batch | `256` / `64` (verl-specific abstraction) | n/a in NeMo-RL | NeMo-RL uses one gradient update per step over `train_global_batch_size` trajectories (with grad accumulation across `train_micro_batch_size`-sized chunks) |
| `train_global_batch_size` | n/a | `510` | matches `num_prompts_per_step × num_generations_per_prompt` (upstream convention; see grpo_math_1B.yaml's `32×16=512`) |
| `train_micro_batch_size` | n/a | `4` (1× and 2× A100 — same; DDP doesn't reduce per-GPU memory) | between upstream's tested `1.5B@seq=512@micro=4` and `8B@seq=4096@micro=1`; with `activation_checkpointing: true` should fit. Drop to 2 if OOM. |
| GRPO group size G | `5` | `5` (`num_generations_per_prompt`) | match |
| KL coef (β) | `0.001` | `0.001` (`reference_policy_kl_penalty`) | match |
| **KL estimator** | `low_var_kl` (Schulman 2020 k3) | `k3` (NeMo-RL default) | **byte-identical formula**; see [`VERL_REFERENCE.md`](VERL_REFERENCE.md) §2 |
| Clip ratio (ε) | `0.2` | `0.2` (`ratio_clip_min=ratio_clip_max=0.2`) | match |
| Max sequence length | `4096` | `4096` (`max_total_sequence_length`) | match |
| Max response length | `500` | `500` (`generation.max_new_tokens`) | match |
| **Max obs length** | `500` tokens per `<information>` block | `2000` chars (`env.search_r1.max_obs_chars`) | **char proxy** — pure-Python parsers, no tokenizer; ~4 char/token for English Wiki text |
| Rollout temperature / top-p | `1.0` / `1.0` | `1.0` / `1.0` | match |
| Max search turns | `4` (`max_turns=4`) | `4` (`env.search_r1.max_turns`, `grpo.max_rollout_turns`) | match |
| **State masking** | `True` — mask `<information>` from policy gradient | automatic via role-based `token_loss_mask` ([grpo.py:1685-1693](../../training/nemo_rl/nemo_rl/algorithms/grpo.py#L1685-L1693)) | **equivalent**, no config knob; env emits `role: tool` → loss=0 |
| GAE λ / γ | `1` / `1` | n/a — GRPO uses leave-one-out, not GAE | paper Appendix lists these but PPO-only |
| Optimizer / Precision | Adam / bf16 | AdamW / bf16 | match (AdamW = Adam with decoupled weight decay; same gradient direction) |
| Gradient checkpointing | enabled | enabled (`activation_checkpointing: true`) | match |
| FSDP CPU offload | enabled (verl-side memory budget tight on 8× H100) | off (`dtensor_cfg.cpu_offload: false`) | DTensor on A100 80GB has more headroom; flip to `true` if OOM persists |
| GPU memory utilization | `0.6` (verl) | `0.6` on 1× A100 / `0.7` on 2× A100 | conservative first-run defaults; raise after observing W&B `gpu_monitoring` |

## 7. Compute

> **Numbers are smoke-anchored** (2026-05-02 1× A100 80GB SXM run, 20 traj/step, mean ~57 s/step) and extrapolated linearly + sub-linearly to the real config. Source: [`SMOKE_RESULTS_2026-05-06.md` "Timing, GPU utilization, and bottlenecks"](SMOKE_RESULTS_2026-05-06.md#timing-gpu-utilization-and-bottlenecks). The "TBD" / "high uncertainty" framing of earlier revisions is gone — we have the per-step measurement and the only remaining unknown is how reward / format-validity evolves over hundreds of steps.

| | Paper (8× H100) | Paper config on 1× A100 | Ours — measured + extrapolated (1× A100) | Ours — extrapolated (2× A100) | Ours — extrapolated (1× H100) | Ours — extrapolated (1× H200) |
|---|---|---|---|---|---|---|
| Hardware | 1 node × 8× H100 | 1× A100 80GB | 1× A100 80GB SXM | 2× A100 80GB SXM | 1× H100 80GB SXM | 1× H200 141GB SXM |
| Training steps | 1005 | 1005 | 1005 | 1005 | 1005 | 1005 |
| Prompts / step | 512 | 512 | 102 | 102 | 102 | 102 |
| Generations / prompt | 5 | 5 | 5 | 5 | 5 | 5 |
| Trajectories / step | **2560** (512 × 5) | **2560** | **510** (102 × 5) | **510** | **510** | **510** |
| Total trajectories | **~2.57M** (= 2560 × 1005) | ~2.57M | **~513k** (= 510 × 1005 = 512,550) | ~513k | ~513k | ~513k |
| Total prompts seen | ~515k (= 512 × 1005 = 514,560) | ~515k | ~103k (= 102 × 1005 = 102,510) | ~103k | ~103k | ~103k |
| Epochs over 169,615-row corpus | **~3.03×** (= 514,560 / 169,615) | ~3.03× | **~0.604×** (= 102,510 / 169,615) | ~0.604× | ~0.604× | ~0.604× |
| Gradient updates / step | 10 (verl `ppo_mini_batch_size=256` over 2560 trajectories) | 10 (gbs=256) | 1 (gbs=510 == prompts × gen → one optimizer.step()) | 1 | 1 | 1 |
| Total gradient updates | ~10,050 | ~10,050 | ~1,005 | ~1,005 | ~1,005 | ~1,005 |
| Per-step time, linear est. | n/a | ≈ 25.5 × 57 s × 5 ≈ 121 min | **24 min** (= 25.5 × 57 s) | 14 min | 12 min | 10 min |
| Per-step time, sub-linear est. | n/a | ~75 min | ~15 min | ~9 min | ~7 min | ~6 min |
| **Wall-clock / run** | ~24 h | **~1300–2000 h** (55–85 d) | **17 d / 11 d** = 264–408 h | **9.5 d / 6.5 d** = 156–228 h | **8.5 d / 5 d** = 120–204 h | **7 d / 4 d** = 96–168 h |
| GPU-hours / run | ~192 GPU-h | 1300–2000 GPU-h | 264–408 GPU-h | 312–456 GPU-h | 120–204 GPU-h | 96–168 GPU-h |
| Vast on-demand $/hr (typical, ±50%) | n/a | ~$1.20 | ~$1.20 | ~$2.40 (2× rate) | ~$2.00 | ~$2.80 |
| **$ / run** | n/a | **$1600–2400** (= 1300 × $1.20 to 2000 × $1.20) | **$300–490** (= 264 × $1.20 to 408 × $1.20) | **$370–550** (= 156 × $2.40 to 228 × $2.40) | **$240–410** (= 120 × $2.00 to 204 × $2.00) | **$270–470** (= 96 × $2.80 to 168 × $2.80) |
| Recommended for Phase 2 | n/a | **infeasible** (>$1.5k/run, 55–85 d) | viable but slow | solid alternative | **best $/run** | fastest, marginal $$ |

### Derivation

**Smoke anchor.** Mean per-step time on 1× A100 80GB SXM at 20 trajectories/step was **~57 s** (8 step measurements across 4 combos, geometric mean of ratios; raw table at [`SMOKE_RESULTS_2026-05-06.md` "Per-step wall-time"](SMOKE_RESULTS_2026-05-06.md#per-step-wall-time-smoke-shape-20-trajectoriesstep)). Real config is 510 trajectories/step = 25.5× the smoke shape.

**Linear extrapolation (upper bound).** Per-step time scales ~linearly with trajectory count when the GPU is generation-bound (rollouts dominate ~34% of the per-step trace; the remaining 66% is logprobs + training + retrieval-wait + weight-refit which scale with batch size too). 25.5 × 57 s = **1453 s ≈ 24.2 min/step**. × 1005 steps = **24,303 min = 405 h ≈ 16.9 days ≈ 17 d**.

**Sub-linear extrapolation (lower bound).** Larger micro-batches utilise the GPU better, but with `policy.sequence_packing.enabled: false` (Qwen3.5 GatedDeltaNet kernel crashes with packed sequences; see [`training/fix/CHANGES.md`](../../training/fix/CHANGES.md) §5) the gain is modest. Empirical heuristic: ~15 min/step. × 1005 = **15,075 min = 251 h ≈ 10.5 d ≈ 11 d**.

**Cost.** 1× A100 80GB SXM on Vast.ai medians ~$1.20/GPU-h. `264 × $1.20 = $317`; `408 × $1.20 = $490`. So `$300–490 / run`. Same arithmetic produces the other hardware columns; cited in the rightmost table column.

**Paper config on 1× A100 (hypothetical).** Run the paper's exact batch sizes (`num_prompts_per_step=512`, `gbs=256` → 10 gradient updates / step) on our hardware. Rollout produces 2560 trajectories per step (5× current); training does 10 gradient updates per step. Per-step time scales ~5× from the rollout blowup (rollout dominates wall-clock); training overhead is modest because gradient accumulation already chunks the work. Net: 5× the 264–408 h baseline = **~1300–2000 h ≈ 55–85 days per run**. (Earlier "250–750 h / 10–30 d" was a rough estimate before the smoke baseline existed — keeping the qualitative point: this is infeasible on 1× A100. Cross-ref: [`docs/edu/BATCH_MATH.md`](../edu/BATCH_MATH.md).) The cheap alternative for closing the **gradient-update** gap (without the rollout blowup) is `gbs=51` on the current 102-prompt config — 10 updates/step at no extra rollout cost; epoch coverage stays at 0.604×.

**Multi-hardware extrapolation.** Cross-vendor relative throughput: H100 ≈ 2× A100 bf16 prefill at this batch size; H200 ≈ 1.2× H100; 2× A100 ≈ 1.7× single A100 wall-clock once the vLLM ⇄ DTensor split eliminates the colocation swap (the largest single idle gap on 1× A100 — [`SMOKE_RESULTS_2026-05-06.md` "Bottlenecks identified"](SMOKE_RESULTS_2026-05-06.md#bottlenecks-identified) #2). All four columns derived directly from these factors against the 1× A100 anchor.

### First-run gate

Wall-clock projection is now smoke-anchored (not "TBD"), so the first-run gate is mostly obsolete; one remaining live decision is whether to commit to 1× A100 or pay the H100 premium up front:
- **Already-decided hardware.** 1× H100 80GB SXM if Vast supply has it (~$400/run, 5–8.5 d). Else 1× A100 80GB SXM (~$300–490/run, 11–17 d).
- **Step-100 health-check.** Regardless of hardware, at step 100 (~1 h H100 / ~2.5 h A100) check W&B `train/reward_mean` curve. If flat at ~0 (no rewardable trajectories) → abort and debug, don't burn the rest of the run.

**Phase-2 budget reality.** ~$1000 USD total → **2–3 full runs**, not the 6 of the original 3-seed × 2-variant plan. The recipe-ablation pivot in [`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md) reflects this.

## 8. Data

| | Paper | Ours |
|---|---|---|
| Training corpus | NQ-train + HotpotQA-train (mixed) — released as [`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) (170k train + 51.7k test) | **Identical source**; prompt rewritten during conversion to use Qwen3.5's `<tool_call>` template (the dataset's `prompt[0].content` ships with paper's `<search>` tags) |
| Format | parquet (verl's expected format) | parquet → NeMo-RL row format (mapping pinned in [`TRAINING_DATA.md`](TRAINING_DATA.md)) |
| Retrieval index | E5-base-v2 + Wiki-18 FAISS Flat IP | **IVF4096-SQ8** (`wiki18_100w_e5_ivf4096_sq8.index`, ~16 GB RAM) — Flat IP times out under training rollout HTTP load; IVF-SQ8 is 3–10× faster. M1 eval still uses Flat IP for paper-fidelity. |

## 9. Divergence summary (one place to glance)

Knowing departures from the paper, in priority order:

1. **Model family**: Qwen3.5-2B (vs. Qwen2.5-3B). Forced by NeMo-RL's Qwen3.5 support. Net: probably comparable; Qwen3.5's stronger base mostly offsets the 1B param drop.
2. **Chat template**: Qwen3.5 native `<tool_call>` (vs. paper's `<search>`). Deliberate; rationale in [`CHAT_TEMPLATE.md`](CHAT_TEMPLATE.md).
3. **Variant naming**: "hybrid" (= Qwen3.5-2B with soft-switch reasoning) replaces "instruct". The Qwen3.5 family does not ship an Instruct variant.
4. **Hardware**: 1×–2× A100 80GB (vs. 8× H100). Wall-clock will be longer; results should be unchanged.
5. **RL framework**: NeMo-RL (vs. verl). Core algorithm (GRPO with leave-one-out advantage, KL to ref, clip 0.2, β 0.001) is identical; engineering details (DTensor vs. FSDP) differ.

Everything else — reward function, group size, learning rate, warmup, batch sizes, sequence lengths, sampling temperature, training steps, validation set — is matched to the paper.

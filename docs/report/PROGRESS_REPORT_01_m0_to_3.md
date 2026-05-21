---
title: Progress Report 01
tags: [report, supervisor, m3, m3.1]
source: internal
created: 2026-05-05
updated: 2026-05-08
---

Leiden University
Leiden Institute of Advanced Computer Science (LIACS)

# Progress report 1

Gaurisankar Jayadas

| | |
|---|---|
| **Date:** | 2026-05-04 |
| **Subject:** | Resource-constrained post-training of small LMs for search-augmented multi-hop QA: Search-R1 reproduction and pivot to Qwen3.5-2B on NeMo-RL |
| **Mentor:** | Prof. dr. Aske Plaat |
| **Co-supervisor:** | Dr. Álvaro Serra Gomez |
| **Project term:** | February 2026 – July 2026 |
| **E-mail:** | gaurisankarj1996@gmail.com |
| **Window covered:** | 2026-04-23 – 2026-05-08 |

## TL;DR

- **Phase-1 (29 ALICE Qwen3-0.6B runs, Apr 3 – Apr 19)**: GRPO + the paper-faithful ReSearch reward (format + EM partial credit) is stable on the hybrid checkpoint, but the 0.6B base cannot bootstrap tool-use cold, the partial-credit reward floor at 0.1 masks the tool-use signal, and prompt phrasing dominates behaviour more than the reward. (Phase-1 builds on the ReSearch paper, not Search-R1; the `<search>` / `<result>` tag scheme is ReSearch's.)
- **M3 (2026-05-07)**: 1046 GRPO steps lifted average EM **0.102 → 0.155 (+52 % relative, +0.053 absolute)** across all 7 paper benchmarks at full Plan A (51,713 items / variant). Held-out 6 / 7 datasets confirms a learned tool-use skill, not memorised answers. Eval pipeline is now pinned and reusable for Phase-2. Checkpoint published: [`pantomiman/Qwen3-0.6B-v0`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0).
- **M3.1 (2026-05-08)**: second checkpoint `el6s2d2h` (no-example + decision-rules prompt, the **highest-reward Phase-1 run** at end-reward 0.215 / +43 % rel over 2280 steps) lifted simple-mean EM further to **0.169** (+9 % rel over M3, +66 % over pre-GRPO); ACC / F1 widen the M3.1-vs-M3 gap to +12 % / +14 %. The no-example variant is a **pareto improvement** (~½ the response budget per episode at higher EM, 40 %-faster eval wall-clock). Confirms that the Phase-1 structural finding (decision-rule scaffolding can substitute for the few-shot example) survives held-out evaluation; recipe transfer to Qwen3.5 is lower-risk. Checkpoint published: [`pantomiman/Qwen3-0.6B-v0.1`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0.1).
- **Pivot**: Phase-1 z7kcxfof stopped at 1046 / 9968 steps after **23 h 47 m 30 s** on 1× A100-40GB (W&B); full horizon ≈ **~9.5 d / run** at the observed pace, and none of the 29 Phase-1 runs reached the full horizon (across the 9 hybrid prompt-ablation runs the projected range is ~5–10 d). Phase-2 Qwen3.5-2B Vast-smoke 57 s/step → **~11–17 d / run on A100-80GB at our affordable 0.6-epoch budget** (1005 steps × 102 prompts/step ≈ 0.604 epochs; the paper's 3-epoch schedule at our batch shape would be ~5× → ~55–85 d / run). Reframed the RQ to *"is the recipe feasible under realistic constraints?"*, targeting the **Qwen3.5 small-model family released 2026-03-02** (0.8B, 2B, 4B, 9B; budget fits 0.8B + 2B); Phase-2 will **start with Qwen3.5-0.8B**. Proposed recipe: E2H + S-GRPO + MC-GRPO with a JustRL plain-GRPO control.

## Outcome of previous meeting

- Run a Search-R1 [1] evaluation baseline analysis: re-evaluate the published `Qwen2.5-3B-Search-R1` checkpoints on the paper's seven QA datasets so we have a clean numerical anchor for downstream training comparisons.
- Keep the original project framing (Plan B), extending RLVR to non-verifiable domains via tool-use as a surrogate objective, as the working scaffold for the thesis.

## Results during the last weeks

- **Phase-1 Qwen3-0.6B training synthesised.** Two training blocks completed earlier on the ALICE cluster (1× A100) were aggregated. v0 (Apr 3 to Apr 9, 14 runs, paper `<search>` / `<result>` tags) tested base vs. hybrid plus a nine-prompt ablation on hybrid varying rules verbosity and the few-shot example. v1 (Apr 12 to Apr 19, 15 runs, in-distribution `<tool_call>` tag format) tested three new prompts on hybrid plus five fresh base-model cold-start attempts. Headline findings:
  - **Base model cannot follow the tool-use format from cold start.** Across the five v1 base-model runs (longest 2300 steps), tool-call rate stayed at 0 throughout. Without instruction tuning or an SFT warm-start, GRPO does not bootstrap the structured format on a 0.6B base. Decision: focus on the hybrid checkpoint.
  - **Hybrid does learn, slowly.** All nine prompt-ablation runs on hybrid showed monotonic reward gains; final means clustered at 0.18 to 0.22 over up to 2300 steps. Training is numerically stable on 1× A100-40GB at the constrained sequence budget.
  - **Prompt drives behavior more than reward.** With training config and reward held constant, prompt phrasing alone moved end-of-run tool-call rate from 0 to 2 per episode and response length from ~480 to ~2050 tokens; reward only moved ±3 pp across the same prompts.
  - **The paper's partial-credit reward creates a floor at 0.1** for any well-formatted but wrong answer. Even runs that abandoned tool use entirely finished at 0.16 mean reward; tool-using runs at 0.18 to 0.22. The 3 to 6 pp gap between tool-using and no-tool behaviors is too small to clearly drive learning.
  - **The `<tool_call>` in-distribution tag format costs nothing at equal step count.** v1's best instruct prompt at 884 steps reached reward 0.179, matching v0's mid-pack at the same step count; v0's higher peak (0.215) just reflects 2.6× more training steps.

- **Search-R1 evaluation baseline reproduced within paper margin.** No eval pipeline was provided upstream; one was built from scratch and locked after a 3-fix audit (`apply_chat=True` for base, restored prompt example sentence, removed runtime `add_special_tokens`). 1000-row stratified subsamples for the five large datasets (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki), plus full Bamboogle (125) and full MuSiQue (2417); single seed, greedy decode. Results across the seven datasets: base **0.292** EM (paper 0.312, Δ −2.0 pp), instruct **0.361** EM (paper 0.336, Δ +2.5 pp); within ±5 pp of paper on 6/7 datasets. Format-validity (`</answer>` close-rate) ≥99.6 % on base and ≥91.4 % on instruct.

- **Pivot to Qwen3.5-2B on NeMo-RL.** Qwen3-0.6B training is too slow on 1× A100 to support reward-function ablation in the remaining timeline, so training pivoted to Qwen3.5-2B (closer to paper's Qwen2.5-3B in capacity, same hybrid soft-switch reasoning toggle). `verl` does not support Qwen3.5, so the Search-R1 training pipeline was ported to NeMo-RL (NVIDIA's GRPO/RLVR framework with Qwen3.5 support): custom Ray-actor environment for the retriever, byte-identical port of the paper's reward function with 15 parity tests passing, and unchanged paper hyperparameters.

- **Wall-clock observed from Vast.ai 1× A100 80GB smoke.** \~57 s/step for 20 trajectories. Two distinct projections to keep separate:
  - **Affordable budget** (1005 steps × 102 prompts/step ≈ **0.604 epochs** of the 169,615-row train corpus, vs. the paper's **3.03 epochs** at 512 prompts/step): ~11 to 17 days per run on 1× A100 80GB (\~\$300 to \$490 / run on Vast); ~5 to 8.5 days on 1× H100 80GB SXM (\~\$240 to \$410); ~6.5 to 9.5 days on 2× A100 (\~\$370 to \$550). This is what the recipe-ablation plan fits into.
  - **Paper-equivalent (3-epoch) training at our batch shape**: ~5× more steps → **~55 to 85 days per run on 1× A100 80GB** (\~\$1,600 to \$2,400). The paper's own batch shape (512 prompts × 5 generations = 2560 traj/step, 5× our 510) on 1× A100 lands in the same envelope. Either way, paper-faithful training is infeasible on our budget.

- **Reframed research question.** Original RQ1 to RQ4 (domain expansion approaches, reward-function modeling, meta-reasoning, curriculum-based training) require a sweep that the remaining compute budget cannot afford. New framing: *"Is it feasible to post-train a small LM to Search-R1-level results under realistic resource constraints, and what is the optimised training recipe?"* The artifacts already produced (prompt-sensitivity, partial-credit floor, reproduced eval baseline, ported NeMo-RL pipeline) become the primary evidence; answerable with one or two trained runs rather than a sweep.

- **First evaluation of the Phase-1 v0 GRPO checkpoint vs the untrained Qwen3-0.6B hybrid (M3, completed 2026-05-07).** The only Phase-1 run that converged on heavy-tool behaviour (`p1_basic_w_ex_z7kcxfof`, 1046 GRPO steps, 2 search / 4 turns; ALICE 1× A100-40GB; 2026-04-06) was evaluated against the untrained Qwen3-0.6B hybrid on the seven paper QA benchmarks. Eval ran on **full Plan A test/dev sets** (51,713 items / variant) rather than 1k subsamples; `sample_num` is not respected by the FlashRAG `search_r1` pipeline path. Hardware: ALICE 1× A100-80GB; greedy decode; single seed.
  - **Result**: average EM **0.102 → 0.155** (+0.053 abs, +52 % rel); 6 / 7 datasets improved (2WikiMultiHopQA tied at −0.003). Single-hop datasets (NQ, TriviaQA, PopQA) gained +69–71 % relative; multi-hop saturated (HotpotQA +40 %, 2Wiki tied) → 0.6 B is **capacity-bound** for multi-hop, not training-bound.
  - **Held-out generalisation rules out memorisation**: `z7kcxfof` was trained on MuSiQue only; the lift transfers to the other 6 benchmarks (none seen at training time), so the model exhibits a learned tool-use skill rather than retrieving training-set answers.
  - Setup cost (14 fixes), wall-clock (~2.5 h / variant), per-dataset breakdown, vs-ReSearch comparison, and the *"eval pipeline now pinned for Phase-2"* deliverable: `docs/report/RESULTS_m3.md` and `docs/report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md` § 4.

- **M3.1 closed (2026-05-08)**. Evaluated the **highest-reward Phase-1 run**, `p3_decide_no_ex_el6s2d2h` (no example, decision rules, end reward 0.215, +43 % rel; vs M3's z7kcxfof at 0.190, +28 % rel) on the same 7 benchmarks at full Plan A. **Result**: simple-mean EM **0.169** vs M3's 0.155 (+0.014 abs, +9 % rel; +0.067 abs / +66 % rel vs pre-GRPO baseline 0.102). 5 / 7 datasets improve over M3, 1 ties, 1 regresses (bamboogle, N = 125, on the edge of small-N noise); biggest wins on PopQA (+0.058), TriviaQA (+0.037), HotpotQA (+0.028). ACC / F1 widen the M3.1-vs-M3 gap to +12 % / +14 %, suggesting the no-example variant produces higher-quality answers that don't always pass EM's strict-match window. The training-curve comparison panel (built from archived W&B histories) shows the structural divergence visually: same base + algorithm + reward + data, only the prompt differs — yet z7kcxfof anchors at heavy-tool 2-call / 4-turn / ~2050-tok behaviour while el6s2d2h converges to standard 1-call / 3-turn / ~1100-tok at slightly higher reward. The no-example variant is a *pareto improvement* (lower compute, higher quality). **Conclusion**: the no-example + decision-rules prompt is a real lever, not just a partial-credit-floor artifact; recipe transfer to Qwen3.5 is lower-risk. Pipeline change was additive only (new `P3_DECIDE_NO_EX_TEMPLATE` + `QWEN3_TEMPLATES` registry; `prompt_mode` family switched to prefix-match). Three sbatch attempts: 2134645 + 2134663 failed on infra waits (cold-cache cliffs at SGLang and retriever `/health`); 2150167 completed in 1 h 32 m on `node870` after wait-window bumps in `scripts/sbatch_m3.sh`. Checkpoint published to HF: [`pantomiman/Qwen3-0.6B-v0.1`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0.1) (parallel to v0). Detail: `docs/milestone_3/MILESTONE_3.1.md`; full numbers + training-curve panel: `RESULTS_m3.md` §14.

## Plans for next weeks

The proposed training recipe is three drop-in additions to a single Search-R1 GRPO baseline run on a Qwen3.5 small model (the family released 2026-03-02 includes 0.8B, 2B, 4B, 9B; only 0.8B and 2B fit the budget). Phase-2 will **start with Qwen3.5-0.8B** for cheap iteration before extending to 2B if the recipe holds. All three additions were chosen because they target the "1× A100 budget" reframing and are cheap to add.

- **E2H data-level curriculum** [2]: train NQ (1-hop) → HotpotQA (2-hop) → MuSiQue (3-hop), ~300 steps per stage, fading out earlier stages. Proven that vanilla GRPO fails on hard tasks for 1.5 to 3B LLMs and that curriculum scheduling recovers them across multiple domains. Cost to add: data scheduler only, no GRPO code change.
- **S-GRPO** [3]: compute the policy-gradient loss on 30 to 50 % of tokens per rollout (informativeness-sampled). Proven on Qwen2-1.5B + LoRA: vanilla GRPO produced 0 gain over the base model; S-GRPO reached +24 pp on SVAMP. Roughly 2× backprop saving per step. Cost: one change in the loss path.
- **MC-GRPO** [4]: replace the mean advantage baseline with the group median (G+1 rollouts, exclude pivot from backprop). Proven to close the G=2 to G=8 quality gap to within 1 % across model families. Cost: one-line baseline swap.

Combined, these are the candidate answer to *"what is the optimised training recipe?"*. They stack on a single GRPO baseline run, fit on 1× A100, and are testable within 2 to 3 affordable runs.

**Required control.** Per JustRL [5], stacking curriculum / length-penalty / dynamic-sampling tricks on top of plain GRPO can *degrade* OOD performance by collapsing exploration. A plain-GRPO control will be run alongside the recipe; if the minimal control beats the stack, the stack is hurting.

## References

[1] *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning.* arXiv [2503.09516](https://arxiv.org/abs/2503.09516), 2025.

[2] Y. Lin et al. *Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning.* arXiv [2506.06632](https://arxiv.org/abs/2506.06632), 2026.

[3] *Token-Efficient RL for LLM Reasoning (S-GRPO).* arXiv [2504.20834](https://arxiv.org/abs/2504.20834), 2026.

[4] *MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout RL.* arXiv [2601.22582](https://arxiv.org/abs/2601.22582), 2026.

[5] *JustRL: Simple RL Recipe for 1.5B Models.* arXiv [2512.16649](https://arxiv.org/abs/2512.16649), 2025.

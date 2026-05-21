---
title: COMPARISON PLAN B SUSPECTS
tags: []
source: internal
created: 2026-04-28
updated: 2026-05-01
---

# Plan B v0 — historical suspect ranking and next-step plan

This is the snapshot of the suspect ranking and recommended next-steps section of [COMPARISON_PLAN_B_v0.md](COMPARISON_PLAN_B_v0.md) as it stood before the v1 sweep convergence (2026-04-28). Kept here as the historical reasoning trail. **All hypotheses below are now resolved**; see [MILESTONE_1.md](../milestone_1/MILESTONE_1.md) and [PAPER_VS_OURS_AUDIT.md](../eval/PAPER_VS_OURS_AUDIT.md) for the converged state.

## Historical: what was *not* the same / could deviate

Ranked by how much it could explain the −8 pp base-variant gap.

### High-suspicion (likely material)

1. **`apply_chat=False` on base** — the load-bearing miss. [PAPER_VS_OURS_AUDIT.md D1](../eval/PAPER_VS_OURS_AUDIT.md#d1-in-detail-the-load-bearing-one) shows verl's `rl_dataset.py:128` applies the chat template unconditionally when the tokenizer has one, and Qwen2.5-3B base ships with a chat template. The paper's reported base GRPO numbers therefore use chat-wrapped prompts. Our `scripts/run_one.sh:35` hard-codes `apply_chat=False` for base. The Bamboogle probe confirmed +1.6 pp lift and 84 %→100 % close-rate from flipping this. **Status: fixed and confirmed on NQ-1k v1 (+7.4 pp from v0).**

   *(Note: an earlier version of this section listed `temperature=1.0, top_p=0.95` as suspect #1, citing the upstream `ppo_trainer.yaml` rollout block. That was incorrect — see [TEMPERATURE_HYPOTHESIS_WRONG.md](TEMPERATURE_HYPOTHESIS_WRONG.md): verl's `_validate()` at `ray_trainer.py:478,508` hard-codes `do_sample=False` and `vllm_rollout.py:162-171` overrides to `temperature=0, top_p=1.0`. The paper eval is greedy. Our `temperature: 0.0` is correct.)*
2. **Base-variant trace hygiene.** Per the smoke profile: instruct hits `</answer>` 125/125, base length-truncates 21/125 on Bamboogle. Plan B does not record `format_valid` / `length_truncated` rates per-dataset — we are blind to whether base is silently failing on NQ/TriviaQA/PopQA/HotpotQA the same way. If `step_limit=500` is biting on factoid prompts, the base variant is being undercounted. Action: pull the `search_r1_format_valid` / `search_r1_extracted_answer` fields out of the existing run JSONs and tabulate truncation rate per (dataset, variant). **Status: still open — see MILESTONE_1.md.**
3. **Single seed at greedy.** Greedy decoding makes seed irrelevant in principle, but tokenizer/index nondeterminism + KV-cache effects at temp=0 can still produce ~1–2 pp run-to-run drift on n=1 k. With 5 seeds (Plan A) or 3 seeds (Plan C-lite) the per-dataset SE drops by √5 ≈ 2.2× and ±10 pp gaps stay ±10 pp. **Status: addressed by Plan A.**

### Medium-suspicion

4. **1 k subsample SE on multi-hop.** ~1.5 pp on factoid; bigger (~2.5–3 pp) on 2Wiki/HotpotQA where answer entropy is higher. Doesn't explain the base bias direction but adds noise to per-dataset deltas.
5. **SGLang launch flags.** Currently launching with `--disable-radix-cache --disable-overlap` (per `EVAL_OPS.md:89`). These should not change EM at temp=0, but worth confirming with an A/B because they are a divergence from the upstream SGLang defaults.
6. **Bamboogle instruct overshoot (+12.8 pp).** Already analysed in [REPRODUCIBILITY.md](../eval/REPRODUCIBILITY.md). On n=125 with single seed this is ~3.4 σ out — not within noise, but the explanation (cleaner `</answer>` stop on instruct) is consistent with the same mechanism that drags base down. Same root cause, opposite signs across variants.

### Low-suspicion (already verified or improbable)

7. Index type — using Flat IP, recall=100 %; the IVF-SQ8 alternative loses <1 % recall and would only narrow the gap, not widen it.
8. Tokenizer / chat template handling — already verified byte-identical against upstream for the prompt template, and instruct numbers track the paper closely, which exercises the chat-template path.
9. Paper-version drift — Search-R1 has 5 arxiv revisions; we are comparing to v5 / Table 3. Earlier revisions report slightly different numbers, but not by 8 pp.

## Historical: recommended next steps before Plan A

Ordered by cost / information ratio (as written before the v1 convergence; left here for historical context):

1. **(in flight — highest priority, confirmed lift on Bamboogle) Re-run base variant on NQ-1k and TriviaQA-1k with `apply_chat=True`.** Bamboogle probe above closed the gap to paper exactly via this single flag flip and pushed format-validity 84 %→100 %. NQ/TriviaQA are where the −10 to −16 pp gaps live; if the same mechanism is at play, this is the fix. Modify `scripts/run_one.sh:35` to set `apply_chat=True` for base, or invoke `run_eval.py` directly with `--apply_chat True --generator_model search_r1_base_model` and a save_note like `search_r1_base_applychat_seedN`. **Status: done.**
2. **(5 min) Add the missing prompt sentence** `For example, <answer> Beijing </answer>.` back to `flashrag/search_r1/templates.py` — see [PAPER_VS_OURS_AUDIT.md](../eval/PAPER_VS_OURS_AUDIT.md#prompt-template-micro-divergence-negligible). Free; LOW severity. **Status: done.**
3. **(5 min) Remove the `add_special_tokens` block** in `flashrag/pipeline/active_pipeline.py:37-42` — see [PAPER_VS_OURS_AUDIT.md D8](../eval/PAPER_VS_OURS_AUDIT.md#d8-special-token-additions). Smoke-test that `</search>` still matches as a stop string. **Status: done.**
4. **(30 min, free) Pull format-validity and length-truncation rate from the rest of the existing Plan B base JSONs**, per dataset. Use `'</answer>' in final_response` as the close-rate signal. Anywhere the rate is <90 % is a candidate for the apply_chat fix above. **Status: still open; tracked in MILESTONE_1.md.**
5. **(2 h) Re-run base variant on NQ-1k with `step_limit` raised** (500 → 1024) and check whether truncation rate drops and EM rises. Rules in/out the per-step cap as the cause. Only run if (1) leaves a residual gap. **Status: not needed — apply_chat + minor fixes closed the gap.**
6. **(4 h) One-seed full-NQ base run** (Plan C slice for one dataset, ~3 h on a 4090 with current flags). If full-NQ base EM is also ~0.32 (vs paper 0.421), the gap is not subsample noise and is reproducible at scale — Plan A on base would just confirm the same wrong number. **Status: still open as base + instruct full-NQ; tracked in MILESTONE_1.md.**
7. **(1 day) Add a 2nd and 3rd seed to Plan B** to bracket per-dataset variance before committing to 5-seed Plan A. Cheap insurance. **Status: superseded by Plan A.**
8. **Only after the above resolves the base gap to within ~3 pp on at least one dataset:** launch Plan A. **Status: NQ-1k v1 closed gap to 3.1 pp — gate met.**

Do **not** change `temperature` or `top_p`. Paper eval is greedy ([PAPER_VS_OURS_AUDIT.md D3](../eval/PAPER_VS_OURS_AUDIT.md)). Our config is correct on this axis.

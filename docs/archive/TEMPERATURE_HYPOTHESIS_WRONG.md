---
title: TEMPERATURE HYPOTHESIS WRONG
tags: []
source: internal
created: 2026-04-28
updated: 2026-05-01
---

# Post-mortem: "paper eval uses temperature=1.0" was wrong

**Date**: 2026-04-28
**TL;DR**: I (Claude) hypothesized the Search-R1 paper's reported numbers were obtained at `temperature=1.0`, citing paper Appendix B.2 and the upstream verl `ppo_trainer.yaml` rollout block. The hypothesis was wrong — paper eval is **greedy** (`temperature=0`). Our existing `temperature: 0.0` in `basic_config.yaml` is correct. Acting on the wrong hypothesis (`temp=1.0, top_p=0.95`) would have *diverged* from the paper, not converged.

## What I claimed

In [../archive/COMPARISON_PLAN_B_v0.md](../milestone_1/../archive/COMPARISON_PLAN_B_v0.md) (since corrected) and [CLAUDE.md](../../claude/CLAUDE.md) (since corrected), the suspect ranking briefly listed:

> **Sampling temperature: 0.0 vs paper/upstream.** […] Paper Appendix B.2 specifies *"temperature of 1.0 and a top-p value of 1.0"*; upstream `verl/trainer/config/ppo_trainer.yaml` `actor_rollout_ref.rollout` uses `temperature=1.0, top_p=0.95, top_k=-1, do_sample=True`, with **no separate `val_kwargs`** so the paper's eval ran at these same values.

The proposed action was a two-line config flip in `basic_config.yaml` and a 2-h re-run on NQ-1k + TriviaQA-1k.

## What I missed

The verl framework hard-codes `do_sample=False` for the validation pass *separately* from the YAML defaults:

- `verl/trainer/ppo/ray_trainer.py:478,508` — `_validate()` injects `do_sample=False` into `meta_info` before calling the rollout worker.
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py:162-171` — when `do_sample=False`, **overrides** the YAML `temperature`, `top_p`, `top_k`, `n` to `(0, 1.0, -1, 1)` regardless of what the rollout config says.

So the rollout YAML block (`temperature=1.0, top_p=0.95`) describes **training rollouts only**. Validation reuses the same vLLM engine but with the override flags. There is no separate `val_kwargs`; the override is hard-coded inside `vllm_rollout`.

Authoritative reading: [PAPER_VS_OURS_AUDIT.md D3](../eval/PAPER_VS_OURS_AUDIT.md) (D3 is the corrected reading of this question, written 2026-04-28).

## What went wrong in my reasoning

1. **I read the rollout YAML and `train_grpo.sh`, then stopped.** Those were the visible "configuration surface" of sampling. I treated them as the full picture.
2. **I assumed "no `val_kwargs` block in the YAML" meant "validation reuses the YAML defaults."** That assumption is wrong for verl — the override path lives in code, not config.
3. **I wrote the paper-quote ("temperature 1.0 and top-p 1.0") into the same suspect bullet as the upstream-config quote** without checking whether the paper sentence describes training rollouts or eval. Appendix B.2 specifically says "rollout sampling" — which in RL parlance is the *training* rollout, not eval. I should have flagged the ambiguity instead of conflating the two.
4. **I didn't grep the upstream repo for `_validate` or `do_sample`** before claiming the eval path. A 30-second `grep -rn 'do_sample' verl/` would have caught it.

## What I should have done

Given the Search-R1 paper does not specify an explicit eval temperature anywhere (confirmed by re-reading), the right move is to trust the **eval code path** of the upstream repo. Concretely:

1. Find the eval entry point (`verl/trainer/ppo/ray_trainer.py:_validate()`).
2. Trace what kwargs it passes to the rollout worker.
3. Trace how those kwargs interact with the rollout YAML defaults — do they override or layer?

That's what the user's [PAPER_VS_OURS_AUDIT.md](../eval/PAPER_VS_OURS_AUDIT.md) does, exhaustively, for every sampling knob. It's the pattern to imitate.

## Correct picture (per the audit)

| Knob | Paper claim | Upstream eval (verl) | Ours | Status |
|---|---|---|---|---|
| `temperature` | rollout 1.0; eval not stated | **0** (override when `do_sample=False`) | 0.0 | **MATCH** |
| `top_p` | rollout 1.0; eval not stated | 1.0 (override) | 1.0 (default) | **MATCH** |
| `top_k` | not stated | -1 (override) | -1 (default) | **MATCH** |
| `do_sample` | not stated | **False** | implicit False | **MATCH** |
| `n` | not stated | 1 (forced when `do_sample=False`) | 1 | **MATCH** |

The actually load-bearing divergence on the base variant is `apply_chat=False` ([D1](../eval/PAPER_VS_OURS_AUDIT.md#d1-in-detail-the-load-bearing-one)), not sampling.

## Lessons for future audits

- **Read the validate/eval entry point of the framework, not just the rollout config.** Training-time configs frequently get overridden at eval-time.
- **When a paper says "rollout sampling uses X"**, do not assume that applies to evaluation. RL papers use "rollout" specifically for the policy-training loop. Eval is a separate code path.
- **If the upstream repo has no separate `val_kwargs` block, that is not evidence** that eval reuses the rollout config. It's evidence that either (a) the override lives in code, or (b) eval was forgotten by the framework's config schema. In verl's case, (a).
- **Whenever we change a sampling parameter, check both the rollout YAML *and* the validate path** — the validate path is what produced the published numbers.

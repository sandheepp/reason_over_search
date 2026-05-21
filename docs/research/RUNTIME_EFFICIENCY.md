---
title: RUNTIME EFFICIENCY
tags: []
source: internal
created: 2026-05-03
updated: 2026-05-06
---

# Runtime Efficiency — Systems & Engineering Levers

**Companion to [`PARADIGM_REVIEW.md`](PARADIGM_REVIEW.md).** That doc covers the *algorithmic* literature (KL, Dr. GRPO, DAPO, LoRA, spec-dec). This one covers the *systems* knobs: vLLM config, optimizer kernels, colocation cost, dynamic batching, fused AdamW, gpu_memory_utilization, prefix caching. The two compose — engineering wins are mostly orthogonal to algorithm wins.

**How to use**: If you want a clear decision path through all three research rounds + both systems/algorithms tracks, see [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md). It cross-references both PARADIGM_REVIEW and this doc, explains why recommendations evolved, and provides a checklist before running.

**Status**: v1, drafted 2026-05-03. Grounded in the measured per-step numbers from [`SMOKE_RESULTS_2026-05-06.md` "Full-training wall-clock + cost"](../training/SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config).

---

## 0. Baseline (measured, 1× A100 80GB SXM)

From the smoke run scaled up to the real config (510 trajectories/step, 1005 steps):

| Hardware | Per-step | 1005 steps | Reference |
|---|---:|---:|---|
| **1× A100 80GB SXM** | **15–24 min** | **11–17 days** | [SMOKE_RESULTS_2026-05-06.md "Full-training wall-clock"](../training/SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config) |
| 1× H100 80GB SXM | 7–12 min | 5–8.5 days | same |
| 2× A100 80GB SXM | 9–14 min | 6.5–9.5 days | same |

**Phase share** (qualitative, from smoke-run profile + co-location swap pattern):

- **Rollout (vLLM gen + retrieval + KV setup): ~50–65% of step.** Multi-turn dominates because every turn needs a fresh prefill of the appended retrieval block.
- **Logprob passes (current + reference): ~15–25% of step.**
- **Train fwd+bwd+optim: ~20–30% of step.** With activation checkpointing on (current default), the recompute pass is non-trivial.
- **Colocation swap (vLLM ⇄ DTensor mode flip, ~35 GiB freed each step): single largest idle gap.** Called out as bottleneck #2 in [SMOKE_RESULTS_2026-05-06.md "Bottlenecks identified"](../training/SMOKE_RESULTS_2026-05-06.md#bottlenecks-identified).

**MuSiQue caveat**: train data is currently NQ+HotpotQA ([`grpo_qwen3.5_2b_1xa100.yaml:376`](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml#L376)). MuSiQue is harder (3-hop, lower base EM → more zero-gradient groups), so wall-clock estimates above are a *floor* if you swap in MuSiQue. Mitigation is on the algorithmic side — see PARADIGM_REVIEW §6 (sample-efficient training) and §5 (dynamic sampling).

---

## 1. Lever map (where each change buys you time)

| # | Lever | Targets | Est. speedup | Risk | Algorithmic counterpart |
|---|---|---|---:|---|---|
| **R1** | vLLM prefix caching | rollout | **1.5–2.0×** on rollout | very low | — |
| **R2** | vLLM async engine | rollout | 1.3–1.5× on rollout | low | — |
| **R3** | Async GRPO (overlap rollout/train) | total step | 1.3–1.6× on step | small staleness, bounded by ε=0.2 | — |
| **R4** | Bump `gpu_memory_utilization` 0.6→0.85 | rollout | 1.2–1.4× on rollout | OOM; touchy under colocation | — |
| **R5** | Raise `generation_batch_size` 32→64–96 | rollout | 1.1–1.3× on rollout | OOM (paired with R4) | — |
| **R6** | Retrieval LRU cache (query-keyed) | rollout | 1.1–1.3× on retrieval (~10% of rollout) | none | — |
| **R7** | EAGLE-3 spec-dec | rollout | 1.4× rollout, no acc delta | low | **PARADIGM_REVIEW §7** (the canonical writeup; defer there) |
| **C1** | Decolocate vLLM (separate process / 2nd GPU) | colocation swap | 1.15–1.30× on step | needs 2 GPUs OR async-mode rework | — |
| **G1** | Dynamic sampling (drop all-1 / all-0 groups) | wall-clock to convergence | 1.3–1.6× to convergence | none meaningful | **PARADIGM_REVIEW §2 #3, §5** |
| **G2** | Drop KL term (β=0) | step + frees ~4–6 GB | 1.15–1.25× on step | drift → use entropy bonus / overlong shaping | **PARADIGM_REVIEW §3** (full evidence) |
| **G3** | Reduce G=5 → 4 | per-step | 1.2× per step; noisier baseline | medium (may wash out) | **PARADIGM_REVIEW §5** |
| **O1** | `fused: true` on AdamW | optimizer step | 1.03–1.08× on step | none | — |
| **O2** | 8-bit AdamW (bnb) | optim memory | enables O4 | none for RL | — |
| **O3** | LoRA r≥64 (attn+MLP) | logprob + train phases | 2–3× on those phases | **capacity ceiling at <2B** | **PARADIGM_REVIEW §4** — Plasticity vs Rigidity says r<256 fails on reasoning at ≤1.5B; treat as VRAM rescue, not free win |
| **O4** | Drop activation ckpt + raise micro-batch 2→4–6 | train phase | 1.3–1.5× on train | OOM unless O2 or O3 frees memory | — |
| **O5** | `torch.compile` on policy | train phase | 1.10–1.25× on train | 5–10 min compile cost; occasional graph breaks | — |
| **O6** | Fix sequence packing | train + logprob | 1.3–1.6× on those phases | blocked by Qwen3.5 GDN kernel bug ([CHANGES.md §5](../../training/fix/CHANGES.md)); only viable if you swap to a non-GatedDeltaNet model | — |
| **M1** | Cap `max_total_sequence_length` 4096→3072 | rollout + train | 1.10–1.20× | truncation rate goes up | — |
| **M2** | Train fewer steps (early-stop on val) | wall-clock | **1.5–3×** | requires re-enabling validation ([VALIDATION.md](../training/VALIDATION.md)) | — |
| **M3** | Move retriever off the A100 | frees VRAM for R4 | 1.05–1.15× | needs IVF index on CPU or 2nd GPU | — |

`fp8 kv cache` is omitted: skip on Ampere (A100 has no native FP8; emulation gives ~0).

---

## 2. Rollout phase — biggest target

Rollout is 50–65% of step time and almost entirely systems-bound. The wins here compound multiplicatively because R1, R2, R4, R5 all improve different bottlenecks (prefix dedup, scheduling, KV memory headroom, batch size).

### R1. Prefix caching (`vllm_kwargs.enable_prefix_caching: true`)

Highest-value rollout knob you don't currently have on. Every group of 5 generations shares:
- The system prompt (~200–400 tokens) — verbatim across all 510 trajectories in a step.
- The user prompt (~200–500 tokens) — verbatim across all 5 in each group.
- Turn-1 retrieved docs (~300–800 tokens) — usually identical across all 5 in a group, since the model often emits the same first search query.

That's typically 700–1700 cached tokens per group of 5, vs ~5× the prefill work without caching.

Config:
```yaml
policy.generation.vllm_kwargs:
  enable_prefix_caching: true
```

### R2. Async vLLM engine (`vllm_cfg.async_engine: true`)

Continuous batching across in-flight requests instead of static `generation_batch_size=32` waves. Big for multi-turn workloads where requests finish at different turns and bunch up.

### R3. Async GRPO

Already wired in the config ([`grpo_qwen3.5_2b_1xa100.yaml:70-74`](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml#L70-L74)) but disabled. Overlaps rollout for step N+1 with the policy update of step N.

```yaml
grpo.async_grpo:
  enabled: true
  max_trajectory_age_steps: 1
  in_flight_weight_updates: false   # leave off until R1+R2 are stable
```

The clip ratio ε=0.2 bounds the off-policy correction. One step of staleness is well-studied as harmless at this clip range.

### R4 + R5. VRAM headroom for vLLM

Current `gpu_memory_utilization: 0.6` is conservative — chosen for safety on the first run ([config comment, line 346–350](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml#L346-L350)). Once W&B `gpu_monitoring` shows the actual peak, push to 0.80–0.85. With more KV cache headroom, raise `generation_batch_size` 32 → 64 (or 96 if W&B confirms headroom).

The constraint is co-location: the training side needs to grab back ~35 GiB during the optimizer step ([SMOKE_RESULTS_2026-05-06.md "Bottlenecks identified"](../training/SMOKE_RESULTS_2026-05-06.md#bottlenecks-identified)). If you decolocate (C1), you can push utilization to ~0.92.

### R6. Retrieval LRU

`retriever_serving.py` already batches within a turn. Add a process-local LRU keyed on the search query string. Same query recurs:
- Within a group of 5 (model often emits identical first-turn queries).
- Across epochs of the same prompt.

Cache size of ~50k entries is a few hundred MB, no big deal.

### R7. Spec-dec

See [`PARADIGM_REVIEW.md` §7](PARADIGM_REVIEW.md#7-rollout-side-savings) for the full evidence base. The headline: NeMo-RL native EAGLE-3 buys 1.41× on rollout with no accuracy delta on 8B reasoning. Effort is 1–2 days config. Integrate **after** R1+R2+R4 so you measure the delta on top of a tuned baseline.

---

## 3. Colocation swap — the hidden tax

[SMOKE_RESULTS_2026-05-06.md "Bottlenecks identified"](../training/SMOKE_RESULTS_2026-05-06.md#bottlenecks-identified) flags this as the #2 bottleneck: each step the `CuMemAllocator` releases ~35.72 GiB to swap modes between vLLM (rollout) and DTensor (training). The swap itself is the largest idle gap in the per-step trace.

Three options:

1. **2-GPU split** (`run_grpo_2xa100.sh` exists): vLLM on GPU 0, DTensor on GPU 1. Eliminates the swap. ~1.5–1.7× total wall-clock vs 1× A100 — best $/run if Vast supply has 2× A100 boxes ([SMOKE_RESULTS_2026-05-06.md "Full-training wall-clock"](../training/SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config)).
2. **Async GRPO with `in_flight_weight_updates: true`**: amortizes the swap across overlapping rollout/train. Combine with R3.
3. **Stay colocated, accept the cost**: today's config. Cheapest hardware, slowest wall-clock.

Option 1 is the cleanest engineering fix; option 2 is the "do more with the same GPU" path.

---

## 4. Algorithm-side speedups (defer to PARADIGM_REVIEW)

Three items have direct wall-clock impact and are covered exhaustively in `PARADIGM_REVIEW.md`. Summarized here for completeness; do **not** redesign these from this doc — read the cited section first.

| Lever | Wall-clock effect | Read |
|---|---|---|
| **Drop KL (β=0)** | Saves one full ref-policy forward per step (~1.15–1.25× on step) **and** frees 4–6 GB VRAM that R4/R5 can reclaim | [PARADIGM_REVIEW §3](PARADIGM_REVIEW.md#3-removing-or-replacing-the-kl-term) |
| **DAPO dynamic sampling** | Fewer wasted steps (gradient zero on all-1/all-0 groups) → 1.3–1.6× faster *to convergence* (per-step time ~unchanged) | [PARADIGM_REVIEW §2 #3, §5](PARADIGM_REVIEW.md#5-grpo-alternatives-and-simplifications) |
| **Dr. GRPO normalization fix** | Per-step time unchanged; better gradient quality → fewer steps | [PARADIGM_REVIEW §5](PARADIGM_REVIEW.md#5-grpo-alternatives-and-simplifications) |
| **LoRA r≥64** | 2–3× on logprob+train phases **but** capacity ceiling — see Plasticity vs Rigidity caveat | [PARADIGM_REVIEW §4](PARADIGM_REVIEW.md#4-lora--qlora-in-rl-post-training) |

The combined recommendation in PARADIGM_REVIEW §11 (Variant B) is: β=0, Dr. GRPO normalizations, DAPO dynamic sampling, EAGLE-3 spec-dec. That's the algorithmic side of the same speedup story this doc tells from the systems side.

---

## 5. Optimizer & training-phase

The training phase is ~20–30% of step. Smaller absolute target than rollout, but the wins are cheap.

### O1. Fused AdamW

Currently `fused: false` and `foreach: false` ([`config:296-297`](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml#L296-L297)). Free win on `optimizer.step()` for 2B params. ~1.03–1.08× on full step — small absolute, zero risk, one config flag.

### O2. 8-bit AdamW (bitsandbytes)

Halves optimizer state memory: 2B × 8 bytes (m + v in fp32) → 4 bytes per param → ~8 GB → ~4 GB. The freed VRAM enables O4 (drop activation checkpointing, raise micro-batch).

NeMo-RL doesn't ship bnb integration for the DTensor backend out of the box; this is a real PR-level effort, not a config flag. Worth scoping if O4 is the goal.

### O3. LoRA

Skip as a first lever — see [PARADIGM_REVIEW §4](PARADIGM_REVIEW.md#4-lora--qlora-in-rl-post-training) for why (Plasticity vs Rigidity: r<256 fails on reasoning at ≤1.5B). Use only as VRAM rescue if R4/O4 still don't fit. If you do, set r=64+ on attn+MLP and budget for a 2–5 EM-point regression vs full FT.

### O4. Drop activation checkpointing + raise micro-batch

Currently `activation_checkpointing: true` and `train_micro_batch_size: 2` ([`config:160-162`](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml#L160-L162)). Recompute eats ~30% of training-pass time. If O2 or O3 frees memory, drop checkpointing and raise micro-batch to 4–6 → 1.3–1.5× on training phase.

### O5. `torch.compile`

NeMo-RL supports it via DTensor. ~1.10–1.25× on fwd+bwd for a 2B model. Long initial compile (5–10 min) is a fixed cost amortized over 1005 steps.

### O6. Sequence packing — blocked

Currently disabled because Qwen3.5's GatedDeltaNet kernel crashes with packed sequences ([config comment, lines 264-270](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml#L264-L270), [CHANGES.md §5](../../training/fix/CHANGES.md)). Would normally give 1.3–1.6× on training + logprob phases. Two paths to unblock:
- Patch the GDN kernel upstream (real work; out of scope unless you become the maintainer).
- Switch base model to **Qwen2.5-1.5B-Instruct** (no GDN). Packing then works. PARADIGM_REVIEW §8's Search-R1 Empirical Study notes general-purpose bases train more stably than reasoning-specialized ones — this swap is defensible on its own.

---

## 6. Misc

### M1. Sequence length cap

Currently `max_total_sequence_length: 4096`. If your rollout token histogram (W&B `gen_tokens` distribution) rarely hits 4k, dropping to 3072 saves ~25% on attention compute and KV memory. **Run a 50-step rollout-only profile first** to confirm — multi-hop MuSiQue is exactly where 4k *might* be the active constraint.

### M2. Fewer steps + early stopping

Cheapest "speedup" of all. The 1005-step number is the Search-R1 NQ/HotpotQA recipe; PARADIGM_REVIEW §6 (1-Shot RLVR, Hard Examples) shows much smaller training budgets work at 1.5B–7B on rule-verified tasks. Re-enable validation ([VALIDATION.md](../training/VALIDATION.md)) with `val_period: 50` and stop when val EM plateaus. Likely converges in 300–500 steps on MuSiQue → 1.5–3× wall-clock.

### M3. Move retriever off the A100

If `retriever_serving.py` is loading FAISS into the same GPU, it's competing with vLLM for VRAM. Move to CPU index (latency hit, may be tolerable since retrieval is <10% of rollout) or a separate small GPU. Frees VRAM for R4.

---

## 7. Suggested ordering

Land in this order — each step's win is measurable in W&B before the next:

1. **One-flag wins** (~1 hour total): R1 (prefix caching), O1 (fused AdamW), R4 (gpu_mem_util 0.6 → 0.75 cautiously). Expect ~1.5–1.8× total step.
2. **Async vLLM** (R2): ~1 day to validate. Stack: 1.8–2.3×.
3. **Bigger gen batch** (R5) + **retrieval LRU** (R6): ~half day. Stack: 2.0–2.7×.
4. **Algorithmic stack from PARADIGM_REVIEW Variant B**: β=0 (G2), Dr. GRPO normalizations, DAPO dynamic sampling. ~3–4 days. Compounds to ~3–4× wall-clock to convergence.
5. **Spec-dec** (R7): 1–2 days config. Stack: 3.5–5×.
6. **Decolocation** (C1) — if you have 2× A100 budget. Otherwise async GRPO (R3).
7. **Drop activation ckpt + raise micro-batch** (O4) — only after one of {O2, O3} frees memory.

Stop measuring before each new lever. The single biggest mistake is stacking five changes at once and being unable to attribute regressions when one of them misbehaves.

**Combined plausible target**: 11–17 days → **3–5 days** wall-clock for a single 1× A100 run, without LoRA, without async-GRPO. With LoRA + async-GRPO accepted: ~1.5–2.5 days, with the algorithmic-quality caveats from PARADIGM_REVIEW §4.

---

## 8. What this doc does *not* cover

- Reward design, sample efficiency, RL algorithm choice → [`PARADIGM_REVIEW.md`](PARADIGM_REVIEW.md).
- Concrete config diffs / scripts to apply each lever → followups; this is the menu, not the recipes.
- MuSiQue data prep (train split isn't in `data/musique/` yet — only `dev.jsonl`). Out of scope here.
- Multi-node training. PARADIGM_REVIEW §10 ("skip these") notes async-RL systems like AReaL assume multi-node — same applies to anything in this doc that mentions "decolocation" beyond a 2-GPU box.

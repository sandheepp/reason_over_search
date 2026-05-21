---
title: HARDWARE_COMPARISON v1 — accelerator + provider comparison (archived 2026-05-11 PM)
tags: [hardware, training, m5, vast, runpod, thundercompute, gpu, archived]
source: internal
created: 2026-05-11
updated: 2026-05-11
status: superseded
supersedes: HARDWARE_COMPARISON_v0.md
---

> **DEPRECATED — superseded 2026-05-11 PM**. This v1 was anchored on the **steps 15–17 mean (10.6 min/step)**, which turned out to be the *trough* of a U-shape. The per-step time bottomed at step 17 and then climbed back up as the model learned to produce longer (better) rollouts; by step 42 it had risen to **~24 min/step**. The full-run estimate on A100 doubled from ~4.5 d / $130 to **~10.2 d / $294**, and the cost-spread that closed in v1 has re-opened — porting to faster hardware is back on the table.
>
> **Live version**: [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md). Narrative and structure below preserved as the v1 baseline for audit purposes; do not act on the numbers in this file.

# Accelerator & provider comparison — M5.1 training (v1, archived PM 2026-05-11)

> **Purpose**: Decide which GPU (and which cloud provider) to run a single M5.1 GRPO training run on. Anchored on **live per-step measurements from the M5.1 production run** (Qwen3.5-0.8B GRPO on NeMo-RL, MuSiQue, 1× A100-80GB on Vast — see [`../report/RESULTS_SMOKE_m5.md` §6](../report/RESULTS_SMOKE_m5.md#6-m51-production-training--live)). Spec table carry-forward from [`../setup/HARDWARE_4090.md`](../setup/HARDWARE_4090.md) (historical 4090 dev-box doc).
>
> **Supersedes** [`HARDWARE_COMPARISON_v0.md`](HARDWARE_COMPARISON_v0.md), which was anchored on the step-8 33 min/step measurement. Per-step time has since collapsed to ~10 min/step (step 17) as the model learned shorter rollouts. The v0 numbers are off by 3× and its recommendation no longer holds; this doc replaces it.
>
> **TL;DR (live trend, 2026-05-11 ~09:30 UTC, step 17)**: at the **new ~10–11 min/step steady-state**, the 1× A100 path lands at **~4.5 d / ~$130** — so the wait-or-swap question is much weaker. Every faster option lands at **$65–150 / 0.6–2 d**. Cheapest fastest: **1× B200 ≈ 14–16 h / ~$90**; cheapest absolute: **2× H100 (TP=2) ≈ 22 h / ~$80**; cheapest single-GPU clean swap: **1× H100 SXM ≈ 2 d / ~$90**. Multi-day savings vs A100 are still real, but cost-deltas are now under $50, so the call hinges on **how much "save 2-3 days of wait" is worth**, not "save $300".

## 1. Live anchor — what we're measuring against

Live trajectory through step 17 (out of 622) from M5.1 production run ([`RESULTS_SMOKE_m5.md` §6](../report/RESULTS_SMOKE_m5.md#6-m51-production-training--live)). Per-step time has dropped 5.6× from step 1 to step 17 as the model learns to write shorter rollouts; the curve is still slightly trending down at step 17 but flattening.

Steady-state anchor: **steps 15–17 mean = 10.6 min/step**. Phase breakdown from step 16 (1× A100-80GB, micro=1 forced):

| Phase | Time / step | Share | Scales with |
|---|---:|---:|---|
| `policy_training` | 7.07 min | **69.8%** | BF16 TFLOPS + mem-BW; **micro=1 forced** by 80 GB VRAM |
| `policy_and_reference_logprobs` | 2.02 min | 19.9% | Mem-BW (fp32 cast `[B,S,V]`); micro=1 forced same reason |
| `generation` (vLLM async) | 0.88 min | 8.7% | Mem-BW only; gen length already collapsed 7×, vLLM async + chunked prefill saturated |
| **Total (steady)** | **~10 min** | — | |
| **Integrated full run** | **~109 h ≈ 4.5 d** | — | Steps 1-17 sunk = ~5.2 h; remaining 605 steps × ~10 min = ~101 h |

**Why the v0 numbers were off**: v0 anchored on step 8 (33 min/step, 1200-token rollouts). At that point the model was still producing near-max-length thinking + multi-turn search loops. By step 17 the model has learned to commit to an answer in ~650 tokens — 3.3× shorter — and the per-step time scales nearly linearly with that. This is *learning dynamics*, not a hardware change; every config below benefits from the same rollout-length collapse. The relative speedups between hardware configs are unchanged from v0, but the absolute baseline shifts 3×.

**The micro=1 bottleneck (unchanged)**: NeMo-RL v0.6.0's TP=1 path casts the full `[B,S,V]=[1,8192,248320]` logits to fp32 before chunking ([`distributed/model_utils.py:1378`](https://github.com/NVIDIA/NeMo-RL)); the chunked path only exists for TP>1. At 80 GB this OOMs at micro=2 by 0.62 GiB. Three unblock paths:
1. **More VRAM** (H200 141 GB, B200 192 GB) → fits trivially; yaml flip.
2. **TP=2** (multi-GPU) → activates the chunked path; same yaml flip.
3. **Source patch** to `model_utils.py:1378` → deferred (M5.3 candidate).

## 2. Accelerator spec reference (carry from `../setup/HARDWARE_4090.md`)

Headline FP16/BF16 dense throughput (no sparsity), memory, interconnect. Real-world throughput depends on kernels, batch size, interconnect — see §3 for per-task M5.1 estimates.

| Accelerator | Arch | Memory | Mem BW | FP16/BF16 (TFLOPS) | FP8 (TFLOPS) | Interconnect | TDP | M5.1 verdict |
|-------------|------|--------|--------|--------------------|--------------|--------------|-----|--------------|
| RTX 4090 | Ada Lovelace | 24 GB | 1.0 TB/s | ~165 | ~330 | PCIe 4 (no NVLink) | 450 W | Too small for M5.1 |
| A100 40 GB | Ampere | 40 GB HBM2e | 1.55 TB/s | 312 | — | NVLink 3 (600 GB/s) | 400 W | Too small (M5.1 needs ≥80 GB) |
| **A100 80 GB** *(live now)* | Ampere | 80 GB HBM2e | 2.0 TB/s | 312 | — | NVLink 3 (600 GB/s) | 400 W | **Baseline — ~4.5 d** |
| H100 PCIe | Hopper | 80 GB HBM3 | 2.0 TB/s | ~756 | ~1513 | PCIe 5 (NVLink bridge opt.) | 350 W | ~2.3 d; ~15% slower than SXM |
| H100 SXM | Hopper | 80 GB HBM3 | 3.35 TB/s | 989 | 1979 | NVLink 4 (900 GB/s) | 700 W | **~2 d**; micro=1 still forced (80 GB) |
| H100 NVL (94 GB) | Hopper | 94 GB HBM3 | 3.9 TB/s | ~835 | ~1670 | NVLink bridge (600 GB/s) | 350-400 W | ~1.8 d; 94 GB may unlock micro=2 (tight) |
| **H200 SXM** | Hopper | 141 GB HBM3e | 4.8 TB/s | 989 | 1979 | NVLink 4 (900 GB/s) | 700 W | **~1.2 d; micro=2 unlocked** |
| **B200** (Blackwell) | Blackwell | 192 GB HBM3e | 8 TB/s | ~2250 | ~4500 / 9000 (FP4) | NVLink 5 (1.8 TB/s) | 1000 W | **~14-16 h; micro=2 unlocked** |
| MI300X | CDNA 3 | 192 GB HBM3 | 5.3 TB/s | ~1300 | ~2600 | Infinity Fabric (896 GB/s) | 750 W | ROCm + NeMo-RL + Qwen3.5 hybrid path = uncharted; **skip** |
| TPU v5p | Google TPU | 95 GB HBM | 2.76 TB/s | 459 (BF16) | — | ICI 3D torus | ~300 W | Requires XLA/JAX port; not available |

Numbers are vendor "dense" (non-sparse) TFLOPS. NeMo-RL is not configured for FP8 on our config, so Hopper/Blackwell FP8 columns are aspirational, not realized in our wall-clock.

## 3. M5.1 wall-clock + cost estimates by hardware

Math anchored on **live 10.6 min/step** (§1, steps 15–17 mean), with three multipliers:

- **Compute/BW speedup vs A100-80GB SXM** (MLPerf + vendor docs, BF16):
  - H100 PCIe ≈ 1.9× (~15% slower than SXM due to lower TDP + PCIe interconnect)
  - H100 SXM ≈ 2.2×
  - H200 SXM ≈ 2.4–2.6× (extra mem-BW helps at vocab=248k cast)
  - B200 ≈ 5–6× over A100 (BF16; MLPerf B200/H100 ≈ 2.2–2.27×, H100/A100 ≈ 2.2×)
- **micro=2 unlock** (only ≥141 GB VRAM, or TP≥2 via chunked path): ~1.5× on training+logprobs phase (≈30% on total step). Same effect from the deferred `model_utils.py:1378` patch.
- **Multi-GPU TP=2 scaling**: ~1.7–1.9× on training+logprobs (good NVLink scaling at this model size); chunked logprobs path comes for free; generation phase doesn't TP-split in vLLM async (conservatively keep single-GPU BW).

| Config | Per-step (steady) | Total run (622 steps) | Code work | Risk |
|---|---:|---:|---|---|
| 1× A100-80GB *(live)* | ~10–11 min | **~4.5 d (~109 h)** | none | live |
| 1× H100-80GB PCIe | ~5.5 min | ~2.3 d | none | clean |
| **1× H100-80GB SXM** | **~4.6 min** | **~2.0 d (~48 h)** | none | clean |
| 1× H100-80GB SXM + chunk patch | ~3.3 min | ~1.4 d | NeMo-RL source mod + smoke | moderate |
| **1× H200-141GB** | **~2.8 min** | **~1.2 d (~29 h)** | yaml `micro=2` | clean |
| 2× A100-80GB (TP=2) | ~4.2 min | ~1.8 d (~44 h) | TP=2 yaml + smoke | moderate |
| **2× H100-80GB SXM (TP=2)** | **~2.1 min** | **~22 h** | TP=2 yaml + smoke re-validate | moderate |
| **1× B200-192GB** | **~1.4 min** | **~14–16 h** | yaml `micro=2`; verify Blackwell sm_100 image | image-compat |
| 2× H200-141GB (TP=2) | ~1.7 min | ~18 h | TP=2 yaml + smoke | moderate |
| 4× H100-80GB SXM (TP=4) | ~1.1 min | ~12 h | larger reconfig | high |

These are **steady-state extrapolations**; if the model's rollout length continues to shrink, all estimates will tighten further (and proportionally for every config). If gen length rises (failure mode), estimates loosen.

## 4. Provider pricing snapshot (May 2026)

Spot/marketplace lows from each provider's pricing page. **Verify before launch — prices move weekly.**

| GPU | Vast.ai | RunPod (spot) | RunPod (Secure) | Thundercompute | Notes |
|---|---:|---:|---:|---:|---|
| A100-80GB SXM | ~$0.79–1.20/h | $1.39/h | ~$1.89/h | **$0.78/h** (proto) | Thundercompute cheapest, but PCIe-only & beta |
| H100-80GB PCIe | $2.00/h | ~$2.39/h | ~$3.39/h | **$1.38/h** (proto) | Thundercompute cheapest PCIe |
| H100-80GB SXM | **$1.87/h** | $2.69/h | ~$3.89/h | not offered | Vast cheapest SXM |
| H100 NVL (94 GB) | rare | rare | ~$3.50/h | not offered | Limited inventory |
| H200-141GB | inconsistent | **$3.59/h** | ~$4.99/h | not offered | RunPod cleanest H200 source |
| B200-192GB | spotty (queue) | **$5.98/h** | ~$7.99/h | not offered | Very limited; expect queue on RunPod |
| MI300X-192GB | rare | ~$2.49/h some | — | not offered | Skip for this run (ROCm risk) |

### Thundercompute caveat — read before committing

Per [Thundercompute pricing](https://www.thundercompute.com/pricing) and [getdeploying.com review](https://getdeploying.com/thunder-compute): Thunder Compute Inc., founded 2024 (Cambridge MA), is a **legitimate but early-stage / beta** provider with the most aggressive A100/H100-PCIe pricing in the market. Important constraints:

1. **PCIe only, no SXM** — for our compute-bound training, H100 PCIe is ~15% slower than H100 SXM (lower TDP + bandwidth). H100-NVLink-bridge variant available; SXM not.
2. **Virtualized GPU** stack (their core IP is GPU-virtualization, see [Thundercompute vs CoreWeave](https://www.thundercompute.com/blog/coreweave-gpu-pricing-review)) — not necessarily a problem for training, but performance under heavy CUDA-IPC workloads (which NeMo-RL relies on for weight-share between Ray actors and vLLM) is **untested for us**. Recommend a 30-min smoke before a multi-day run.
3. **No H200 / B200 / SXM** in their fleet as of May 2026.
4. **"Prototyping" vs "Production" tiers** — production tier carries SLAs and is higher-priced; the $0.78 / $1.38 figures are prototyping. For a 4-day run on prototyping tier the platform may not guarantee dedicated allocation.
5. Per-minute billing; no egress fees; persistent storage $0.05/GB/month.

**Bottom line on Thundercompute for M5.1**: viable as the **cheapest A100 single-GPU swap** at $0.78/h (~$85 for a 4.5 d run, vs $130 on Vast) **if** the smoke passes and we don't mind beta-grade stability for a 2-3 d run. **Not a path to <2 d**, since they don't offer SXM/H200/B200.

### Provider call

- **Vast.ai**: cheapest H100 SXM ($1.87/h) and the SKU breadth (4090 → H100) we already have a runbook for ([`SETUP_VAST.md`](../vast/SETUP_VAST.md)). Host quality varies — fine for short or restartable runs.
- **RunPod Secure Cloud**: 1.3–1.5× the price of Vast spot, but enterprise-grade datacenters → safer for a multi-day run. **Recommended for B200 / H200** (where you want it to *just work*).
- **RunPod Community Cloud (spot)**: middle ground.
- **Thundercompute**: best on $/hr for A100 and H100 PCIe, but PCIe-only, beta, virtualized — smoke-test before committing.

## 5. Cost-per-run table (all configs)

Cost = total wall × $/h. Compare to the "live A100" baseline (~$130).

| Config | Wall | Provider | $/h | **Cost** |
|---|---:|---|---:|---:|
| 1× A100 *(live, doing nothing)* | 4.5 d | Vast | $1.20 | **~$131** |
| 1× A100 (swap to cheapest) | 4.5 d | Thundercompute | $0.78 | **~$85** |
| 1× H100 PCIe | 2.3 d | Thundercompute | $1.38 | **~$76** |
| **1× H100 SXM** | **2.0 d** | **Vast** | **$1.87** | **~$90** |
| 1× H100 SXM | 2.0 d | RunPod (Secure) | $3.89 | ~$187 |
| 1× H200 | 1.2 d | RunPod (spot) | $3.59 | ~$104 |
| 1× H200 | 1.2 d | RunPod (Secure) | $4.99 | ~$145 |
| 2× A100 (TP=2) | 1.8 d | Vast | $2.40 | ~$106 |
| **2× H100 SXM (TP=2)** | **22 h** | **Vast** | **$3.74** | **~$82** |
| **1× B200** | **15 h** | **RunPod (spot)** | **$5.98** | **~$90** |
| 1× B200 | 15 h | RunPod (Secure) | $7.99 | ~$120 |
| 2× H200 (TP=2) | 18 h | RunPod (spot) | $7.18 | ~$129 |
| 4× H100 SXM (TP=4) | 12 h | Vast | $7.48 | ~$90 |

**Striking property**: every config now lands in **$75–150**. The cost-spread that justified "rent B200 to save $300" in v0 is gone. The decision is now **time vs risk**, not **time vs cost**.

## 6. Recommendation for a single M5.1 run

The right call depends on what you optimize for. Three viable archetypes:

### A) "Just let the A100 finish" — do nothing (~4.5 d, ~$130)
- **Why pick this**: training is healthy (reward climbing 0.020 → 0.13, KL stable, generation length collapsing). Any swap has nonzero disruption cost (first checkpoint at step 50 = ~30 h from start; before that, killing the run = restart from scratch).
- **Why not**: 4.5 d of wall-clock has its own cost (delays M5.2 and downstream evals).
- **Verdict**: the **default**. Skip the rest of this doc unless you specifically need the run done in <2 d.

### B) "Single-GPU clean swap to ≤2 d" — 1× H100 SXM on Vast (~2 d, ~$90)
- **Why pick this**: cleanest swap (yaml unchanged, no Blackwell compat smoke, no multi-GPU config rewrite, no patch). Halves wall-clock vs A100 for a similar dollar amount.
- **Why not**: still 2 d. If you need <1 d, see C.
- **Pre-commit**: a 30-min smoke on the H100 SXM box to confirm the existing `pantomiman/reason-over-search-v1:v2` image comes up + per-step time lands as projected. ~$1 spend.

### C) "Single-GPU absolute fastest" — 1× B200 on RunPod Secure (~15 h, ~$120)
- **Why pick this**: same-day finish. 192 GB unlocks micro=2 trivially. RunPod Secure avoids the host-loss failure mode.
- **Why not**: Blackwell sm_100 compat for the v2 worker venv (mamba-ssm / flash-attn / nv-grouped-gemm) is **untested by us**. A 30-min B200 smoke (~$3) answers this; if it fails, fall back to (D).
- **Verdict**: best if you want <1 d **and** are willing to spend 30 min de-risking the image.

### D) "Multi-GPU fallback if B200 smoke fails" — 2× H100 SXM (TP=2) on Vast (~22 h, ~$82)
- **Why pick this**: same-day finish at the cheapest total cost. TP=2 unlocks the chunked-logprobs path → micro=2 for free.
- **Why not**: TP=2 config rewrite + smoke re-validation against M4 byte-exact prompt. Higher process risk than (B) or (C).

### Cost-minimum option — 1× A100 80GB on Thundercompute (~4.5 d, ~$85)
- Cheapest absolute. **Same wall-clock as the live A100 path** (no speedup) — only saves $45 vs status quo. Requires a 30-min smoke on Thundercompute's virtualized stack (NeMo-RL CUDA-IPC weight share is the failure mode to watch for). Not worth the swap unless the live A100 box becomes unstable.

**My single answer**: **Option B (1× H100 SXM on Vast)** is the best Pareto point for this run. Halves wall-clock vs status quo, cleanest swap, $90, no Blackwell risk. Take option C only if <1 d is a hard requirement; take option A if the 4.5 d is fine and you'd rather not touch the live run.

## 7. What I'd not do

- **Don't kill the in-flight A100 run before step 50** (~25 h from start at current pace). First checkpoint lands at step 50; killing earlier loses everything. The step-50 checkpoint is recoverable on any hardware.
- **Skip MI300X**. ROCm + NeMo-RL + vLLM + Qwen3.5 hybrid arch is uncharted; a day burned on bring-up.
- **Skip FP8**. NeMo-RL v0.6.0 doesn't wire Transformer Engine FP8 through DTensor for Qwen3.5; substantial port for one run.
- **Don't pick H100 PCIe over SXM for the same provider**. 15% slower at often ≥50% the SXM price — bad value. PCIe only makes sense at Thundercompute where they don't offer SXM.

## 8. Open questions before launch

- Does NeMo-RL v0.6.0's v2 worker venv come up on Blackwell sm_100? If the `pantomiman/reason-over-search-v1:v2` image was compiled against Hopper-era CUDA only, mamba-ssm / flash-attn / nv-grouped-gemm may need a rebuild. **A 30-min B200 smoke answers this.**
- Does NeMo-RL's CUDA-IPC weight share survive Thundercompute's GPU virtualization? Our M5 smoke v5 already established that `expandable_segments` alloc breaks the `pidfd_getfd` IPC path inside Ray actors ([`RESULTS_SMOKE_m5.md` §1 failure-mode #5](../report/RESULTS_SMOKE_m5.md#11-log-artifact-map)); virtualized GPUs are another layer of indirection that could trip the same wire.
- For multi-GPU (option D): does the M4 byte-exact prompt check still pass at TP=2? The rollout shape doesn't change but the rendering path goes through a different code branch.
- Do we want H200-or-B200 as a permanent training surface (in which case rebuild [`SETUP_VAST.md`](../vast/SETUP_VAST.md) as `SETUP_RUNPOD.md` first), or one-off rental for this run?

## 9. Pointers

- 4090 dev-box historical snapshot: [`../setup/HARDWARE_4090.md`](../setup/HARDWARE_4090.md)
- v0 of this doc (step-8 anchor; superseded): [`HARDWARE_COMPARISON_v0.md`](HARDWARE_COMPARISON_v0.md)
- Live M5.1 trajectory and bottleneck analysis: [`../report/RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md)
- Current Vast setup runbook: [`../vast/SETUP_VAST.md`](../vast/SETUP_VAST.md)
- M5.1 milestone: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- Paper-vs-ours mapping: [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md)
- Phase-2 runtime-efficiency lever menu: [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md)

## 10. Sources (May 2026)

- [Vast.ai GPU Pricing](https://vast.ai/pricing), [Vast H100 SXM](https://vast.ai/pricing/gpu/H100-SXM), [Vast H100 PCIe](https://vast.ai/pricing/gpu/H100-PCIE)
- [RunPod Pricing](https://www.runpod.io/pricing), [RunPod GPU breakdown — Northflank](https://northflank.com/blog/runpod-gpu-pricing)
- [Thundercompute Pricing](https://www.thundercompute.com/pricing), [Thundercompute A100 pricing](https://www.thundercompute.com/blog/nvidia-a100-pricing), [Thundercompute H100 pricing](https://www.thundercompute.com/blog/nvidia-h100-pricing), [getdeploying.com Thundercompute review](https://getdeploying.com/thunder-compute)
- [GPU Cloud Pricing Comparison 2026 — Spheron](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/), [H100 Rental Prices Compared — IntuitionLabs](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [H100 vs A100 BF16 throughput — Spheron](https://www.spheron.network/blog/nvidia-a100-vs-h100/), [bestgpusforai.com](https://www.bestgpusforai.com/gpu-comparison/a100-vs-h100)
- [B200 vs H100 — Civo](https://www.civo.com/blog/comparing-nvidia-b200-and-h100), [Lightly](https://www.lightly.ai/blog/nvidia-b200-vs-h100), [WhiteFiber LLM training infra](https://www.whitefiber.com/blog/choosing-gpu-infrastructure)
- [B200 vs H200 — Northflank](https://northflank.com/blog/b200-vs-h200)

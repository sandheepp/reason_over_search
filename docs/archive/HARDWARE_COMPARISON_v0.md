---
title: HARDWARE_COMPARISON v0 — accelerator + provider comparison (archived 2026-05-11)
tags: [hardware, training, m5, vast, runpod, gpu, archived]
source: internal
created: 2026-05-11
updated: 2026-05-11
status: superseded
---

> **DEPRECATED — superseded 2026-05-11**. This v0 was anchored on step-8 of the M5.1 live run (~33 min/step on A100). By step 17 per-step time had collapsed to ~10 min/step as the model learned shorter rollouts; that cuts the A100 full-run estimate from 14-16 d down to ~4-5 d and shifts every downstream config and cost estimate. The recommendation also shifts: at the new numbers, **most configs (including A100) land at $100-150 and ≤2 d**, so the call is less "fastest" and more "cheapest single-GPU swap".
>
> **Live version**: [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md). The narrative and structure below are preserved as the v0 baseline for audit purposes; do not act on the numbers in this file.

# Accelerator & provider comparison — M5.1 training (v0, archived)

> **Purpose**: Decide which GPU (and which cloud provider) to run a single M5.1 GRPO training run on. Anchored on **live per-step measurements from the M5.1 production run** (Qwen3.5-0.8B GRPO on NeMo-RL, MuSiQue, 1× A100-80GB on Vast — see [`../report/RESULTS_SMOKE_m5.md` §6](../report/RESULTS_SMOKE_m5.md#6-m51-production-training--live)). The accelerator-spec table is the carry-forward from the previous [`../setup/HARDWARE_4090.md`](../setup/HARDWARE_4090.md) (historical 4090 dev-box doc).
>
> **TL;DR (v0, superseded)**: 1–2 days is feasible only on **1× B200** (single-GPU swap, ~1.5–2 d, ~$250) or **2× H100/H200 with TP=2** (~1.5–3 d, multi-GPU config tax). 1× H200 single-GPU lands at ~3.5–4 d. 1× H100 single-GPU lands at ~7 d. The current A100 path is the worst on both wall-clock and $-per-run.

## 1. Live anchor — what we're measuring against

Live trajectory from M5.1 production run, step 8 of 622 ([`RESULTS_SMOKE_m5.md` §6.5](../report/RESULTS_SMOKE_m5.md#65-live-timing-breakdown--steps-8-and-10)). All numbers below scale from this anchor.

| Phase | Time / step | Share | Scales with |
|---|---:|---:|---|
| `policy_training` | 23.5 min | **71.6%** | BF16 TFLOPS + mem-BW; **micro=1 forced** by 80 GB VRAM |
| `policy_and_reference_logprobs` | 6.7 min | 20.5% | Mem-BW (fp32 cast `[B,S,V]`); micro=1 forced same reason |
| `generation` (vLLM async) | 2.3 min | 6.9% | Mem-BW; already collapsed by vLLM async + chunked prefill |
| **Total** | **~33 min** | — | |
| **Total run** | **622 steps × ~33 min** | — | **~14–16 d** (live trend at step 8) |

> **Freshness note (2026-05-11 ~08:30 UTC, step 10)**: per-step time has continued dropping as the model learns shorter rollouts — step 10 landed at **21.3 min** ([`RESULTS_SMOKE_m5.md` §6.2](../report/RESULTS_SMOKE_m5.md#62-per-step-trajectory-live-refresh-as-steps-land)). Latest steady-state estimate is **~21–29 min/step → ~10–13 d** for the full A100 run. All wall-clock and cost estimates in §3 below use the conservative step-8 anchor (33 min); the live A100 path is now slightly cheaper than tabulated, but the **relative speedups vs A100 are unchanged** (the rollout-length collapse helps every GPU equally), so the comparison and recommendation stand.

**The micro=1 bottleneck**: NeMo-RL v0.6.0's TP=1 path casts the full `[B,S,V]=[2,8192,248320]` logits to fp32 before chunking ([`distributed/model_utils.py:1378`](https://github.com/NVIDIA/NeMo-RL)); the chunked path only exists for TP>1. At 80 GB this OOMs at micro=2 by 0.62 GiB. Two unblock paths:

1. **More VRAM** (H200 141 GB, B200 192 GB) → fits trivially; yaml flip.
2. **TP=2** (multi-GPU) → activates the chunked path; same yaml flip.
3. **Source patch** to `model_utils.py:1378` → deferred (M5.3 candidate).

## 2. Accelerator spec reference (carry from `../setup/HARDWARE_4090.md`)

Headline FP16/BF16 dense throughput (no sparsity), memory, interconnect, and typical use. Real-world throughput depends on kernels, batch size, and interconnect — see §3 for per-task M5.1 estimates.

| Accelerator | Arch | Memory | Mem BW | FP16/BF16 (TFLOPS) | FP8 (TFLOPS) | Interconnect | TDP | M5.1 verdict |
|-------------|------|--------|--------|--------------------|--------------|--------------|-----|--------------|
| RTX 4090 | Ada Lovelace | 24 GB | 1.0 TB/s | ~165 | ~330 | PCIe 4 (no NVLink) | 450 W | Too small for M5.1 |
| RTX 5090 | Blackwell (consumer) | 32 GB | 1.79 TB/s | ~210 | ~420 | PCIe 5 (no NVLink) | 575 W | Too small for M5.1 |
| A100 40 GB | Ampere | 40 GB HBM2e | 1.55 TB/s | 312 | — | NVLink 3 (600 GB/s) | 400 W | Too small (M5.1 needs ≥80 GB) |
| **A100 80 GB** *(live now)* | Ampere | 80 GB HBM2e | 2.0 TB/s | 312 | — | NVLink 3 (600 GB/s) | 400 W | **Baseline — ~14–16 d** |
| H100 SXM | Hopper | 80 GB HBM3 | 3.35 TB/s | 989 | 1979 | NVLink 4 (900 GB/s) | 700 W | ~7 d; micro=1 still forced (still 80 GB) |
| H100 NVL | Hopper | 94 GB HBM3 | 3.9 TB/s | ~835 | ~1670 | NVLink bridge (600 GB/s) | 350–400 W | ~6–7 d; 94 GB may unlock micro=2 (tight) |
| H200 SXM | Hopper | 141 GB HBM3e | 4.8 TB/s | 989 | 1979 | NVLink 4 (900 GB/s) | 700 W | **~3.5–4 d; micro=2 unlocked** |
| **B200** (Blackwell) | Blackwell | 192 GB HBM3e | 8 TB/s | ~2250 | ~4500 / 9000 (FP4) | NVLink 5 (1.8 TB/s) | 1000 W | **~1.5–2 d; micro=2 unlocked** |
| GB200 (Grace+B200) | Blackwell + Arm | 384 GB (pair) | 16 TB/s (pair) | ~4500 | ~9000 | NVLink 5 + C2C | ~2700 W | Overkill; pod-only |
| MI300X | CDNA 3 | 192 GB HBM3 | 5.3 TB/s | ~1300 | ~2600 | Infinity Fabric (896 GB/s) | 750 W | ROCm + NeMo-RL + Qwen3.5 hybrid path = uncharted; **skip** |
| TPU v5p | Google TPU | 95 GB HBM | 2.76 TB/s | 459 (BF16) | — | ICI 3D torus | ~300 W | Requires XLA/JAX port; not available |

Numbers are vendor "dense" (non-sparse) marketing TFLOPS. NeMo-RL is not configured for FP8 on our config, so Hopper/Blackwell FP8 columns are aspirational, not realized in our wall-clock.

## 3. M5.1 wall-clock + cost estimates by hardware

Math anchored on live 33 min/step (§1), with three multipliers:

- **Compute/BW speedup vs A100-80GB** (MLPerf + vendor docs, BF16):
  - H100 ≈ 2.2× ([Spheron A100 vs H100](https://www.spheron.network/blog/nvidia-a100-vs-h100/))
  - H200 ≈ 2.4–2.6× (same compute as H100; extra BW helps at our vocab=248k cast)
  - B200 ≈ 5–6× over A100 in BF16 ([MLPerf B200 vs H100 ≈ 2.2–2.27×](https://www.civo.com/blog/comparing-nvidia-b200-and-h100); H100 ≈ 2.2× over A100 → 4.8–5.0×; real-world reports up to 4× H100 → 8× A100 ceiling)
- **micro=2 unlock** (only ≥141 GB VRAM, or TP≥2 via chunked path): ~1.5× on training phase (≈30% on total step). The deferred `model_utils.py:1378` patch achieves the same unlock without changing hardware.
- **Multi-GPU TP=2 scaling**: ~1.7–1.9× on compute (good NVLink scaling at our model size); the chunked logprobs path comes for free.

| Config | Per-step | Total run (622 steps) | Code work | Risk | Cost @ provider |
|---|---:|---:|---|---|---:|
| 1× A100-80GB *(live)* | 33–38 min | **14–16 d** | none | live | ~$400–460 (Vast @ $1.20/h) |
| 1× H100-80GB SXM | ~15–18 min | **~7 d** | none | clean | ~$315 (Vast @ $1.87/h) |
| 1× H100-80GB + chunk-fp32 patch | ~10–12 min | ~5 d | NeMo-RL source mod + smoke | moderate | ~$225 |
| **1× H200-141GB** | **~8–10 min** | **~3.5–4 d** | yaml `micro=2` | clean | **~$300–345** (RunPod @ $3.59/h) |
| 2× H100-80GB (TP=2) | ~6–8 min | ~2.5–3 d | TP=2 yaml + smoke re-validate | moderate | ~$225–270 (Vast 2× $1.87/h) |
| **1× B200-192GB** | **~3–4 min** | **~1.5–2 d** | yaml `micro=2`; verify image on Blackwell sm_100 | image-compat | **~$215–285** (RunPod @ $5.98/h) |
| 2× H200-141GB (TP=2) | ~3–4 min | ~1.5–2 d | TP=2 yaml + smoke | moderate | ~$260–345 |
| 4× H100 (TP=4 / DP×TP) | ~2–3 min | ~1–1.5 d | larger reconfig | high | ~$220–300 |

**Counterintuitive but real**: the faster boxes are also **cheaper per run** — GPU-hour scaling beats hourly-rate scaling. The A100 path is worst on both axes.

## 4. Provider pricing snapshot (May 2026)

Sourced from [Vast.ai pricing page](https://vast.ai/pricing) and [RunPod pricing page](https://www.runpod.io/pricing). Spot/marketplace; on-demand ~1.3–2× higher. **Verify before launch — prices move weekly.**

| GPU | Vast.ai (spot/marketplace) | RunPod (spot) | RunPod (Secure Cloud) | Reliability notes |
|---|---:|---:|---:|---|
| A100-80GB SXM | ~$0.79–1.20/h | $1.39/h | ~$1.89/h | Vast cheapest; host quality varies — fine for short runs, risky for 15 d |
| H100-80GB SXM | $1.87/h | $2.69/h | ~$3.89/h | Vast undercuts; reliability variance is the main downside |
| H100-80GB PCIe | $2.00/h | ~$2.39/h | ~$3.39/h | |
| H100 NVL (94 GB) | rare | rare | ~$3.50/h | Limited inventory |
| H200-141GB | inconsistent listings | **$3.59/h** | ~$4.99/h | RunPod has dedicated tier |
| B200-192GB | spotty (queue likely) | **$5.98/h** | ~$7.99/h | Limited datacenters; expect queue on RunPod, very limited on Vast |
| MI300X-192GB | rare | ~$2.49/h some providers | — | Skip for this run (ROCm risk) |

**Provider call**:
- **Vast.ai**: cheapest A100/H100; reliability is host-by-host. Good for short or restartable runs.
- **RunPod Secure Cloud**: 1.3–1.5× the price of Vast spot, but enterprise-grade datacenters → safer for a 1.5–4 d run you can't easily restart. **Recommend for B200 / H200** (where you want it to *just work*).
- **RunPod Community Cloud (spot)**: middle ground.

## 5. Recommendation for a single M5.1 run

**Primary: 1× B200-192GB on RunPod Secure Cloud** (~$250, ~1.5–2 d). Reasoning:

1. Single-GPU = no NeMo-RL multi-GPU config rewrite (no `policy.parallelism.tensor_model_parallel_size`, no TP=2 smoke re-validate against M4 byte-exact prompt).
2. 192 GB fits `train_micro_batch_size: 2` trivially with massive headroom (the dominant 0.62 GiB OOM evaporates).
3. RunPod Secure Cloud avoids the host-disappears-mid-run failure mode that's catastrophic at 36 h.
4. Counterintuitively **cheapest** of all the ≤2-day options.

**Pre-commit validation** (cost: ~$3, ~30 min): run a [v6-equivalent 10-step smoke](../report/RESULTS_SMOKE_m5.md#2-v6--m5-smoke-pipeline-validation-smoke-shape--success) on the B200 instance before kicking off the full run. This catches Blackwell sm_100 image incompatibility (mamba-ssm / flash-attn / causal-conv1d / nv-grouped-gemm in the v2 DTensor venv). If the smoke fails, fall back to:

**Fallback: 2× H200 (TP=2) on RunPod** (~$300, ~1.5–2 d). The chunked-logprobs path comes for free with TP≥2, so the same micro=2 unlock applies. Pays a multi-GPU config tax but no Blackwell compat risk.

**Cheapest-clean option (if 1–2 d isn't a hard requirement)**: 1× H200 on RunPod (~$320, ~3.5–4 d). Zero code risk, 4× faster than today, single-GPU.

## 6. What I'd not do

- **Don't kill the in-flight A100 run before step 50** (~30 h on the current trending pace). First checkpoint lands at step 50; killing earlier loses everything. Step 50 also gives a real datapoint for confirming the 33-min/step trend.
- **Skip MI300X**. ROCm + NeMo-RL + vLLM + Qwen3.5 hybrid arch is uncharted; a day burned on bring-up.
- **Skip 2× A100**. Only buys ~6 d (per [`RESULTS_SMOKE_m5.md` §4.3](../report/RESULTS_SMOKE_m5.md#43-what-could-change-the-answer)) — neither cheaper nor faster than a single H200/H100.
- **Skip FP8**. NeMo-RL v0.6.0 doesn't have Transformer Engine FP8 wired through DTensor for Qwen3.5; would be a substantial port for one run.

## 7. Open questions before launch

- Does NeMo-RL v0.6.0's v2 worker venv come up on Blackwell sm_100? If the `pantomiman/reason-over-search-v1:v2` image was compiled against Hopper-era CUDA only, mamba-ssm / flash-attn / nv-grouped-gemm may need a rebuild. **A 30-min B200 smoke answers this.**
- Do we want H200-or-B200 as a permanent training surface (in which case rebuild [`SETUP_VAST.md`](../vast/SETUP_VAST.md) as `SETUP_RUNPOD.md` first), or one-off rental for this run?
- For multi-GPU (fallback path): does the M4 byte-exact prompt check still pass at TP=2? The rollout shape doesn't change but the rendering path goes through a different code branch.

## 8. Pointers

- 4090 dev-box historical snapshot: [`../setup/HARDWARE_4090.md`](../setup/HARDWARE_4090.md)
- Live M5.1 trajectory and bottleneck analysis: [`../report/RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md)
- Current Vast setup runbook: [`../vast/SETUP_VAST.md`](../vast/SETUP_VAST.md)
- M5.1 milestone: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- Paper-vs-ours mapping (paper-faithful lock): [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md)
- Phase-2 runtime-efficiency lever menu: [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md)

## 9. Sources (May 2026)

- [Vast.ai GPU Pricing](https://vast.ai/pricing), [Vast H100 SXM](https://vast.ai/pricing/gpu/H100-SXM), [Vast H100 PCIe](https://vast.ai/pricing/gpu/H100-PCIE)
- [RunPod Pricing](https://www.runpod.io/pricing), [RunPod GPU breakdown — Northflank](https://northflank.com/blog/runpod-gpu-pricing)
- [GPU Cloud Pricing Comparison 2026 — Spheron](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)
- [H100 vs A100 BF16 throughput — Spheron](https://www.spheron.network/blog/nvidia-a100-vs-h100/), [bestgpusforai.com](https://www.bestgpusforai.com/gpu-comparison/a100-vs-h100)
- [B200 vs H100 — Civo](https://www.civo.com/blog/comparing-nvidia-b200-and-h100), [Lightly](https://www.lightly.ai/blog/nvidia-b200-vs-h100), [WhiteFiber LLM training infra](https://www.whitefiber.com/blog/choosing-gpu-infrastructure)
- [B200 vs H200 — Northflank](https://northflank.com/blog/b200-vs-h200)

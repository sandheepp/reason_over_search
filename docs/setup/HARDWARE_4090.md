---
title: HARDWARE — 4090 dev box (historical)
tags: [hardware, 4090, historical]
source: internal
created: 2026-05-01
updated: 2026-05-11
---

# Hardware — 4090 dev-box snapshot (historical)

> **Scope**: This doc is the snapshot of the original 4090 development box used through M0–M1. As of M5 (2026-05) the active training surface is **1× A100-80GB on Vast.ai** (and we're evaluating H100 / H200 / B200 for M5.1). For the current accelerator comparison and per-run wall-clock / cost estimates, see [`HARDWARE_COMPARISON.md`](HARDWARE_COMPARISON.md). The accelerator-comparison table below is kept here for historical reference only; the live version lives in `HARDWARE_COMPARISON.md`.

## The 4090 dev box (snapshot 2026-04-27)

| Resource | Spec |
|----------|------|
| GPU | 1x NVIDIA GeForce RTX 4090 |
| VRAM | 24 GB GDDR6X (24,564 MiB) |
| GPU power cap | 415 W (idle ~47 W) |
| CUDA / Driver | CUDA 13.0, driver 580.95.05 |
| CPU | AMD EPYC 7642 (Rome, Zen 2) |
| CPU cores / threads | 48 cores / 96 threads, 1 socket |
| CPU clock | 1.5–2.3 GHz |
| Cache | L1d 1.5 MiB, L2 24 MiB, L3 256 MiB |
| RAM | 503 GiB total (~492 GiB available) |
| Swap | 7.6 GiB |
| NUMA | Single node (0–95) |

### Practical implications
- **Single 24 GB GPU** — can fit a 7B model in fp16 (~14 GB) or 13B in 4-bit; 70B needs offloading or quantization to ≤4-bit + CPU spill.
- **Plenty of host RAM (503 GB)** — comfortable for CPU offload, large index/embedding tables, ZeRO-Offload, or holding entire datasets in memory.
- **EPYC 7642 is generation Zen 2** (no AVX-512, no bf16 instructions on CPU) — heavy CPU FP work is slower than on newer Sapphire Rapids / Genoa.
- **No NVLink, no multi-GPU** — tensor/pipeline parallelism is not possible here; stick to single-device training/inference.
- **PCIe-only** — host↔device bandwidth is the bottleneck for any offloading scheme.

## Accelerator Comparison

Headline FP16/BF16 dense throughput (no sparsity), memory, interconnect, and typical use.

| Accelerator | Arch | Memory | Mem BW | FP16/BF16 (TFLOPS) | FP8 (TFLOPS) | Interconnect | TDP | Notes |
|-------------|------|--------|--------|--------------------|--------------|--------------|-----|-------|
| **RTX 4090** *(this box)* | Ada Lovelace | 24 GB GDDR6X | 1.0 TB/s | ~165 (330 w/ sparsity) | ~330 | PCIe 4.0 x16 (no NVLink) | 450 W | Consumer; great single-GPU dev box |
| RTX 5090 | Blackwell (consumer) | 32 GB GDDR7 | 1.79 TB/s | ~210 | ~420 | PCIe 5.0 (no NVLink) | 575 W | Newer consumer top-end |
| A100 40 GB | Ampere | 40 GB HBM2e | 1.55 TB/s | 312 | — (no FP8) | NVLink 3 (600 GB/s), PCIe 4 | 400 W | Workhorse for 2020–2023 training |
| A100 80 GB | Ampere | 80 GB HBM2e | 2.0 TB/s | 312 | — | NVLink 3 (600 GB/s) | 400 W | Same compute, 2× memory |
| H100 SXM | Hopper | 80 GB HBM3 | 3.35 TB/s | 989 | 1979 | NVLink 4 (900 GB/s) | 700 W | FP8 + Transformer Engine |
| H100 NVL | Hopper | 94 GB HBM3 | 3.9 TB/s | ~835 | ~1670 | NVLink bridge (600 GB/s) | 350–400 W | LLM inference SKU |
| H200 SXM | Hopper | 141 GB HBM3e | 4.8 TB/s | 989 | 1979 | NVLink 4 (900 GB/s) | 700 W | Same compute as H100, big memory bump |
| B200 (Blackwell) | Blackwell | 192 GB HBM3e | 8 TB/s | ~2250 | ~4500 (FP8) / 9000 (FP4) | NVLink 5 (1.8 TB/s) | 1000 W | 2024+ training flagship |
| GB200 (Grace+B200) | Blackwell + Arm CPU | 384 GB HBM3e (per pair) | 16 TB/s (pair) | ~4500 | ~9000 (FP8) | NVLink 5 + C2C | ~2700 W | Superchip module |
| MI300X | CDNA 3 | 192 GB HBM3 | 5.3 TB/s | ~1300 | ~2600 | Infinity Fabric (896 GB/s) | 750 W | AMD's H100/H200 competitor |
| **TPU v4** | Google TPU | 32 GB HBM | 1.2 TB/s | 275 (BF16) | — | ICI 3D torus (2.4 Tbps/link) | ~200 W | Pod-scale (4096 chips) |
| **TPU v5e** | Google TPU | 16 GB HBM | 0.82 TB/s | 197 (BF16) / 393 (INT8) | — | ICI 2D torus | ~170 W | Cost-optimized inference/fine-tune |
| **TPU v5p** | Google TPU | 95 GB HBM | 2.76 TB/s | 459 (BF16) | — | ICI 3D torus, 4800 Gbps/link | ~300 W | Training flagship; pods up to 8960 chips |
| **TPU v6e (Trillium)** | Google TPU | 32 GB HBM | 1.6 TB/s | ~918 (BF16) | ~1836 (INT8) | ICI 2D torus | — | 2024+ generation |
| AWS Trainium2 | Custom | 96 GB HBM | 2.9 TB/s | ~650 | ~1300 | NeuronLink | ~500 W | AWS-only |

> Numbers are vendor "dense" (non-sparse) marketing TFLOPS; real-world throughput depends heavily on kernels, batch size, and interconnect.

## Quick mental model

- **VRAM** dictates *what fits*. 4090's 24 GB ≈ A100-40's lower half; an A100-80 / H100 / H200 unlocks 70B-class models without aggressive quantization.
- **Memory bandwidth** dictates *inference token/s* for memory-bound LLM decode. H100 (3.35 TB/s) is ~3.3× the 4090; H200/B200 widen the gap further.
- **FP8 / FP4** matters only on Hopper+ and Blackwell — Ampere and the 4090 cannot do FP8 in hardware (the 4090 *can* do FP8 GEMM but lacks the Transformer Engine integration; Ampere/A100 cannot at all).
- **TPUs** shine *at pod scale* with XLA/JAX. A single TPU v5p chip ≈ A100 in raw TFLOPS but with much faster ICI fabric and no PCIe/NVLink bottleneck — they beat GPUs on cost-per-FLOP when you can fill the pod.
- **Single 4090** is excellent for prototyping, retrieval/embedding workloads, and ≤13B inference; for 70B+ training you want H100/H200/B200 or TPU v5p pods.

## Implications for this repo (`reason_over_search`)

- 24 GB is enough for typical 7B Search-R1-style policies in bf16 with short contexts; for 13B+ you'll need 4-bit (bitsandbytes/AWQ) or activation offloading.
- 503 GB host RAM means the **retriever index can live in RAM** — wiki dumps, FAISS / dense indices, BM25 stores all fit comfortably.
- Single-GPU only → use gradient accumulation rather than DDP; vLLM / SGLang for serving will saturate the 4090 on small models.

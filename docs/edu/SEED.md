---
title: SEED
tags: []
source: internal
created: 2026-05-02
updated: 2026-05-02
---

# What `--seed` controls in our GRPO pipeline

> Educational deep-dive on RNG seeding — what it locks in, what it doesn't, where in the codebase it actually propagates. Reading this is *not* required to run training; the [Phase 2 runbook](../milestone_2/PHASE_2_RUNBOOK.md) is.
>
> **Background**: if "RNG" is unfamiliar, read [`RNG.md`](RNG.md) first.

## ELI5

A seed is the starting position of a card shuffle. Same starting deck + same shuffle pattern → same hands every time. RNG seed = "the starting deck for the dealer".

- **Without a seed**: every launch is unique (good for averaging, bad for reproducibility).
- **With a seed**: launches with the same seed reproduce; launches with different seeds produce different but valid trajectories.

`--seed N` on our bash wrappers sets `grpo.seed=N`, which then propagates into four places.

---

## Where our seed actually goes

I traced `grpo.seed` through NeMo-RL's code. It hits four RNGs.

[`grpo.py:257`](../../training/nemo_rl/nemo_rl/algorithms/grpo.py#L257) calls `set_seed(grpo.seed)` at the top of training:

```python
def set_seed(seed):
    random.seed(seed)              # Python's stdlib RNG
    np.random.seed(seed)           # NumPy
    torch.manual_seed(seed)        # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch on every CUDA device
```

[`vllm_worker.py:94`](../../training/nemo_rl/nemo_rl/models/generation/vllm/vllm_worker.py#L94) passes the same seed to `vllm.LLM(seed=...)` — vLLM has its own RNG, seeded separately.

So the seed locks in:

### 1. Dataset shuffle order

`data.shuffle: true` in our config means before each epoch the dataloader reshuffles the 169,615 training rows. The shuffle uses `torch`'s RNG, which `set_seed` initialized.

So **seed 42 vs seed 7 → totally different prompt order across the 1005 steps**.

This isn't "random samples for training" in the active-learning sense — every sample is seen ~3 times across 1005 steps × 510 trajectories ≈ 514k trajectories vs 169k unique prompts. Seed determines *the order*, not *which subset*.

### 2. Rollout generation (the big one)

Our config has `temperature=1.0`, `top_p=1.0` — full distribution sampling. For each prompt, vLLM generates `num_generations_per_prompt=5` traces. Each generated token is sampled stochastically:

```
P(token | context) = softmax(logits)
next_token = sample_from(P)   # ← this needs an RNG
```

vLLM's RNG (seeded by `grpo.seed`) decides which token gets picked at each step.

**This is where most of the run-to-run variance comes from.** Same prompt + same model + different seed → genuinely different generated traces, often with different `<search>` queries, different reasoning chains, different answers.

### 3. GRPO group formation

For each prompt, the policy generates 5 traces (group size G=5). All 5 are sampled stochastically — the seed determines *which* 5 traces you get out of the exponentially-many possible ones. The advantage is then the within-group baseline:

```
advantage_i = reward_i - mean(reward_1..reward_5)
```

Different seeds → different 5 traces → different advantages → different gradient direction. The variance cancels out *in expectation* over many steps, but each individual training trajectory is unique.

### 4. PyTorch ops (mostly inactive for us)

`torch.manual_seed` controls anything stochastic in the policy forward/backward:

- **Dropout**: Qwen3.5-2B has dropout = 0 in its released config, so this is a no-op for us.
- **Optimizer state init**: irrelevant — we load from a HF checkpoint, optimizer state starts fresh from gradients only.
- **Stochastic kernels**: none in standard Qwen.

For our specific setup, this RNG is mostly inactive. The big variance sources are #1 and #2.

---

## Why 3 seeds × 2 variants

Same model, same hyperparameters, three different seeds = three independent training trajectories. Each seed produces a final EM number. Reporting `mean ± std` across the 3 tells you:

- **Mean**: best estimate of "what this setup achieves on average".
- **Std**: how much you can trust a single number.

M1 audit found Bamboogle had ~3 pp single-run noise at n=125 (see [`docs/archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md`](../archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md)). Without multi-seed averaging, a 2-pp delta between qwen_native and paper arms could be pure RNG, not signal. With 3 seeds, std-of-means shrinks by √3 ≈ 1.7×, separating signal from noise.

The published Search-R1 paper reports single-seed numbers, which is part of why the paper-vs-our comparison needs multi-seed on our side: to rule out that we're chasing noise.

---

## What the seed does NOT control

A few sources of nondeterminism survive even with a fixed seed — worth knowing for debugging:

- **CUDA kernel non-determinism**: some GPU ops (esp. atomic adds in attention) produce slightly different bit patterns each run, accumulated across millions of FLOPs. Two runs with the same seed on the same hardware will diverge by ~1e-5 in logits within a few steps.
- **Hardware**: same seed on A100 vs H100 → divergent trajectories from step 1 (different kernel implementations).
- **Driver / library version**: cuDNN, vLLM, NeMo-RL, torch upgrades change kernels.

So "seed=42 reproduces run X" only holds bit-exactly if you also pin the hardware, drivers, and library versions. In practice, expect runs with the same seed on the same instance to track each other closely (same trajectory, EM within ~0.5 pp), while runs across different instances drift more.

---

## Concrete: what changes between seed 42 and seed 7

| Step | seed=42 outcome | seed=7 outcome |
|---|---|---|
| Step 1 prompt batch | rows [33912, 891, 14411, ...] (102 of them) | rows [88201, 5512, 167003, ...] |
| First rollout for prompt #1 | `<think>I should look this up.</think><tool_call><parameter=query>SpaceX founder year</parameter>...` | `<think>The question asks about... </think><tool_call><parameter=query>who started SpaceX company</parameter>...` |
| First gradient step | computed on those 510 specific traces | computed on a totally different 510 |
| Step 100 val/accuracy | maybe 0.18 | maybe 0.21 |
| Step 1005 final val/accuracy | 0.41 | 0.39 |

Both are *valid* training trajectories of the same setup. Their average (0.40) plus 1-σ band (~0.01) is the Phase-2 result.

---

## The W&B run name

Our bash wrappers also use the seed (and a UTC timestamp) in the W&B run name:

```
qwen3.5-2b-{variant}-search_r1-{arm}-{1,2}xa100-seed{N}-{UTC-timestamp}
```

e.g. `qwen3.5-2b-base-search_r1-qwen_native-1xa100-seed42-20260502T0838Z`.

The timestamp lets you re-launch the same `(variant, arm, seed)` (e.g. resuming after an OOM) and get a distinct W&B run, while still resuming from the seed-keyed checkpoint dir. See [`run_grpo_1xa100.sh`](../../training/scripts/run_grpo_1xa100.sh).

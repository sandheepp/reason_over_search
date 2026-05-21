---
title: RNG
tags: []
source: internal
created: 2026-05-02
updated: 2026-05-02
---

# RNG — random number generators

> Background concept that [`SEED.md`](SEED.md) assumes you know. Read this first if "what is an RNG?" is the question.

## ELI5

**RNG = Random Number Generator** — the part of the program that produces "random" numbers when something needs them (shuffling a deck, sampling a token, picking dropout positions, etc.).

The catch: computers can't actually produce real randomness on demand. So "random" almost always means **pseudo-random** — a deterministic algorithm that produces a stream of numbers that *looks* random but is fully reproducible if you know the starting state.

```
   seed (a single number)
     │
     ▼
   ┌─────────────────────┐
   │  RNG state machine  │      ┌─→ 0.7234...
   │  state ─→ next state│ ─────┼─→ 0.1856...
   │  state ─→ output    │      ├─→ 0.9912...
   └─────────────────────┘      └─→ 0.0143...
              ↑
              keeps internal state, advances each call
```

Same seed in → same sequence out. That's the whole game.

## Pseudo-random vs "true" random

| | Source | Reproducible? | Used for |
|---|---|---|---|
| **Pseudo-random** | algorithm + seed | yes (given the seed) | ML, simulation, games — anywhere reproducibility matters |
| **True random** | hardware noise (thermal, photon arrival, mouse jitter) | no | cryptography, secure key generation |

ML pipelines exclusively use pseudo-random. The whole point of seeding is reproducibility — if your training run depended on hardware entropy, debugging would be hopeless.

## RNGs are stateful, and there are several of them

A fresh RNG starts at state `f(seed)`. Each `random()` call updates state and returns a number derived from it:

```python
state = f(seed)
def random():
    global state
    state = advance(state)   # different state every call
    return derive(state)
```

This is why **the order of calls matters**. If your code does:

```python
random()  # uses state #1
random()  # uses state #2
```

vs.

```python
random()  # uses state #1
torch.zeros(...)   # somebody else's random() snuck in here?
random()  # uses state #3 instead of #2 if so
```

then identical seeds can produce different downstream numbers if the order of consuming them changes. This is one reason refactoring "deterministic" ML code can subtly break reproducibility.

## Multiple RNGs in our stack

Different libraries each have their **own separate RNG with its own state**. Seeding one doesn't seed the others. In our training run, four+ separate RNGs are in play:

| Library | RNG | What our code does | Where it propagates |
|---|---|---|---|
| Python `random` | one global state | shuffling, sampling | `random.seed(42)` |
| NumPy | global + per-Generator | array ops, HF datasets internals | `np.random.seed(42)` |
| PyTorch CPU | global state | dataloader shuffle, CPU tensor ops | `torch.manual_seed(42)` |
| PyTorch CUDA | per-device state | GPU kernels, dropout | `torch.cuda.manual_seed_all(42)` |
| vLLM | its own state | rollout token sampling | `vllm.LLM(seed=42)` |

NeMo-RL's `set_seed(...)` ([`algorithms/utils.py`](../../training/nemo_rl/nemo_rl/algorithms/utils.py)) handles the first four. vLLM is seeded separately at worker init ([`vllm_worker.py:94`](../../training/nemo_rl/nemo_rl/models/generation/vllm/vllm_worker.py#L94)). [`SEED.md`](SEED.md) walks through where each one matters in our specific pipeline.

If a library's RNG isn't seeded explicitly, it falls back to system entropy → that library's "random" calls become non-reproducible even when everything else is fixed. A common bug source.

## What kind of "random" comes out

When you call `random()` you typically get:

- **A float in [0, 1)** — uniform sample from the unit interval. Useful as a probability comparator.
- **Wrap that to anything else**: integer in `[a, b)`, sample from a categorical distribution, sample from a Gaussian (Box-Muller transform), pick a token from `softmax(logits)` (vLLM's case).

vLLM's rollout sampling at `temperature=1.0, top_p=1.0` is conceptually:

```python
probs = softmax(logits)               # vector over vocab
u = vllm_rng.random()                 # one float in [0,1)
chosen_token = first_index_where(cumsum(probs) >= u)
```

Different `u` (i.e. different RNG state) → potentially different `chosen_token`. That's how seed propagates into "what tokens did the model emit".

## Why pseudo-random is good enough for ML

Modern PRNGs (Mersenne Twister, PCG, Philox) have astronomical periods (length of sequence before repeating: 2^19937 for Mersenne Twister) and pass strict statistical tests. For everything ML cares about — sampling tokens, shuffling data, simulating dropout — they're indistinguishable from "true" randomness.

The reproducibility property (same seed → same sequence) is *the* feature, not a bug. It's how:

- A bug found at step 437 can be re-triggered.
- A 3-seed average gives you variance estimates instead of luck-of-the-draw.
- Two collaborators on different machines can verify they're running the same experiment.

## Where pseudo-random reproducibility breaks

Even with all RNGs seeded identically, two runs can still diverge. The biggest culprits:

- **CUDA kernel non-determinism**: many GPU ops (atomic reductions in attention, scatter-add) produce slightly different bit patterns each call due to non-deterministic thread scheduling. PyTorch has `torch.use_deterministic_algorithms(True)` to force the deterministic versions, but they're slower and not all ops have one.
- **Hardware**: different GPU families (A100 vs H100) use different CUDA kernel implementations → different bit-exact outputs even with the same seed.
- **Driver/library version**: cuDNN, vLLM, and PyTorch ship kernel updates that change numerics by epsilon.
- **Floating-point order of operations**: `(a + b) + c ≠ a + (b + c)` in finite precision. Distributed reductions where the order depends on which worker finishes first → subtle non-determinism.

This is why `seed=42` reproduces "approximately" the same run, not bit-exactly, unless you also pin the hardware, drivers, and library versions. In practice, for ML work, "approximately" means trajectories that track each other to ~5–6 decimal places for the first few steps and gradually diverge as small numerical differences amplify.

## TL;DR for the rest of the docs

- "RNG" in our docs always means a pseudo-random number generator, with state.
- "Seed" is the single number that initializes the state.
- Same seed + same code + same hardware + same library versions → same outputs (within floating-point tolerance).
- Different seeds → different but equally valid runs of the same setup.
- See [`SEED.md`](SEED.md) for what each RNG actually does in our GRPO loop.

# What this training run actually does

A conceptual walkthrough of the GRPO training loop — what the policy is learning, why the config looks the way it does, and what a successful run looks like in W&B. For *how to run* it, see [training/README.md](../../training/README.md). For the *upstream-vs-ours* mapping, see [PAPER_VS_OURS_TRAINING.md](PAPER_VS_OURS_TRAINING.md).

## The goal in one sentence

Teach a 2B-parameter LLM (Qwen3.5-2B) to **interleave reasoning steps with search calls** so it can answer multi-hop factoid questions (NQ + HotpotQA) better than it would zero-shot. The policy learns *when to search, what to query, and how to synthesize results into an answer* — entirely from a single sparse reward (did the final answer match the gold answer?).

This reproduces the **Search-R1** paper's setup, ported from verl onto NeMo-RL.

## The training loop, conceptually

Every GRPO step does this:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Sample 102 prompts from NQ+HotpotQA                          │
│                                                                 │
│ 2. For each prompt, generate G=5 different rollouts             │
│    (sampling temperature > 0 → diverse trajectories per prompt) │
│                                                                 │
│ 3. Each rollout is a multi-turn conversation:                   │
│                                                                 │
│      <think> I need to find X </think>                          │
│      <search> who founded X </search>                           │
│         ↓ HTTP POST to local FAISS retriever                    │
│      <information> top-3 docs </information>                    │
│      <think> the answer is in doc 2 </think>                    │
│      <search> when was X founded </search>     ← turn 2         │
│      <information> ... </information>                           │
│      <answer> 1923 </answer>                                    │
│                                                                 │
│    Up to 4 search turns. <answer>...</answer> ends the rollout. │
│                                                                 │
│ 4. Score each rollout: reward = 1 if EM(answer, gold) else 0    │
│                                                                 │
│ 5. GRPO update on the 102×5 = 510 trajectories                  │
└─────────────────────────────────────────────────────────────────┘
```

## What makes it GRPO (not vanilla PPO)

GRPO's trick: instead of training a separate value/critic network to estimate "how good is this state," it **uses the group of G=5 rollouts from the same prompt as its own baseline**.

For each rollout, the advantage is:

```
A_i = (r_i − mean(r_1..r_5)) / std(r_1..r_5)
```

So if 4 of 5 rollouts for a prompt failed and one got EM=1, that lucky rollout gets a strongly positive advantage and the policy is pushed toward whatever it did. If all 5 got EM=0 or all 5 got EM=1, the advantage is zero — no signal, no update for that prompt.

This sidesteps the classic PPO headache of training a value head on sparse 0/1 rewards. The cost is sample inefficiency: you're paying 5× generation per prompt.

The actual loss is a clipped surrogate (PPO-style):

```
L = − E[ min(ρ·A, clip(ρ, 1−ε, 1+ε)·A) ] + β · KL(π || π_ref)
```

with `ε = 0.2`, `β = 0.001`, and the KL is the low-variance k3 estimator (Schulman 2020) — keeps the policy from drifting too far from the base model.

## What the policy is actually learning

Three skills, all from the same EM reward:

1. **Decompose** — break a multi-hop question into searchable sub-queries. The reward only fires after the final `<answer>`, but credit propagates back to earlier `<search>` decisions through the trajectory's joint probability.
2. **Query formulation** — phrase searches in a way the retriever can answer. Bad queries → irrelevant docs → wrong answer → negative advantage on those tokens.
3. **Stop early** — emit `<answer>` when confident. Each extra turn costs tokens; trajectories that loop unnecessarily get out-competed by sibling rollouts that answered on turn 2.

Crucially the model is **not** learning facts — those come from retrieval. It's learning a *policy over tool use*.

## Why the config looks the way it does

| Choice | Reason |
|---|---|
| Qwen3.5-2B-Base (not Instruct) | Want raw next-token modeling without RLHF artifacts that interfere with the new RL objective |
| LR 1e-6 (very low) | RL on a pretrained model is fragile; high LR collapses the policy in <100 steps |
| KL β = 0.001 | Anchor to base model, but loosely — must allow real behavior change |
| `max_turns = 4` | Most NQ/HotpotQA answers reachable in ≤2 hops; 4 leaves slack without runaway costs |
| `top_n = 3` documents | Enough recall without flooding context with noise |
| `max_obs_chars = 2000` | Prevents one bad retrieval from blowing the 4096 context budget |
| 1005 steps | Matches paper's reported convergence point on this dataset |
| Validation every 100 steps | EM on a held-out 1000-sample slice is the only honest signal — train reward is too sparse to track |

Source-of-truth values live in [training/configs/grpo_qwen3.5_2b_1xa100.yaml](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml). Knob-by-knob commentary in [NEMO_RL_KNOBS.md](NEMO_RL_KNOBS.md).

## What "success" looks like in the W&B dashboard

Three curves to watch:

- **`train/reward_mean`** — should rise from ~0.05 (base model basically guessing) to ~0.40–0.50 by step 1000.
- **`val/accuracy` (EM)** — the real metric; should track train reward but with a 50–100 step lag.
- **`train/grad_norm`** — should stay bounded (clip is at 1.0); spikes here signal an instability that usually precedes policy collapse.

If at step 100 `val/accuracy` is still 0 or `train/reward_mean` is flat, the abort gate fires — likely a template/format bug where the policy never produces parseable `<answer>` tags, not a learning rate issue. See [VALIDATION.md](VALIDATION.md) for the diagnostic playbook.

## What this is *not*

- **Not SFT.** No teacher trajectories, no demonstrations. The model writes its own training data via rollouts.
- **Not preference learning (DPO/RLHF).** No human or model preference labels — pure outcome reward (EM=1/0).
- **Not retrieval training.** The FAISS retriever is fixed; only the policy LLM updates. The retriever is a black-box tool, just like a calculator would be.
- **Not improving raw QA knowledge.** A frozen Qwen3.5-2B *with* this learned search policy will outperform a frozen Qwen3.5-2B with naive prompting — because of *how it uses* the same retriever, not because the model knows more.

That last point is the whole thesis of "reason over search": small models + smart tool use > big models alone, on knowledge-intensive tasks.

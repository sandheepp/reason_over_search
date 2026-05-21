# Reason Over Search

## Papers

1. [ReSearch](https://www.alphaxiv.org/abs/2503.19470)
2. [Search-R1](https://www.alphaxiv.org/abs/2503.09516)

## Milestones

- **Milestone 1** — Baseline [Search-R1](https://www.alphaxiv.org/abs/2503.09516) reproduction. See [docs/milestone_1/MILESTONE_1.md](docs/milestone_1/MILESTONE_1.md).
- **Milestone 2** — Training [Search-R1](https://www.alphaxiv.org/abs/2503.09516) on [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (base + hybrid) via [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). See [docs/milestone_2/MILESTONE_2.md](docs/milestone_2/MILESTONE_2.md).
- **Milestone 3** *(closed 2026-05-07)* — Evaluated [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3) hybrid (pre-GRPO vs. 1046-step `p1_basic_w_ex` GRPO checkpoint) on the 7 paper benchmarks. Headline: average EM 0.102 → 0.155 (+52% relative). See [docs/milestone_3/MILESTONE_3.md](docs/milestone_3/MILESTONE_3.md), [docs/report/RESULTS_m3.md](docs/report/RESULTS_m3.md).
- **Milestone 4** — Qwen3.5-0.8B baseline eval pipeline (M4.1–M4.4: action format, prompt search, full 7-dataset sweep). Untrained floor at hybrid mean EM 0.103 with the `qwen35_terse` prompt. See [docs/milestone_4/MILESTONE_4.md](docs/milestone_4/MILESTONE_4.md).
- **Milestone 5** — Qwen3.5-0.8B GRPO training on the ReSearch-paper recipe (NeMo-RL port). See [docs/milestone_5/MILESTONE_5.md](docs/milestone_5/MILESTONE_5.md).
  - **M5.1** — F1-only reward baseline (the production run). Scaffold: [training_m5_1/](training_m5_1/).
  - **M5.3** — Training-time analysis + speed improvements. See [docs/milestone_5/MILESTONE_5_3.md](docs/milestone_5/MILESTONE_5_3.md).
  - **M5.5** — F1 + 0.1 partial-credit + format-gate reward ablation. Scaffold: [training_m5_5/](training_m5_5/). Design: [docs/milestone_5/MILESTONE_5_5.md](docs/milestone_5/MILESTONE_5_5.md).
  - **M5.6** — EM-only reward ablation. Scaffold: [training_m5_6/](training_m5_6/). Design: [docs/milestone_5/MILESTONE_5_6.md](docs/milestone_5/MILESTONE_5_6.md).
- **Milestone 6** — Literature review + experiment planning for publication. See [docs/milestone_6/MILESTONE_6.md](docs/milestone_6/MILESTONE_6.md).

## Start training

**Trigger**: when the user says **"start training"** (with no further qualification), the agent MUST first gather the target host's system configuration before recommending or running anything. Different hardware → different config YAML → different launch script → different setup path. Picking the wrong combo wastes hours.

### Step 1 — agent asks the user for the system configuration

Ask for, at minimum:

| Field | Why it matters |
|---|---|
| GPU model + count | B300 / B200 / H200 / H100 / A100 / 4090; chooses which config YAML and whether TP is enabled |
| VRAM per GPU | Sets `train_micro_batch_size`, `gpu_memory_utilization`, and whether activation checkpointing is required |
| Host RAM | 8 retriever workers each load ~16 GB index; <150 GB host RAM caps `num_retriever` |
| vCPU count | Caps retriever workers + DataLoader workers |
| Storage (free disk, GB) | Bootstrap needs ≥85 GB for Qwen3.5-0.8B path; ≥130 GB for 2B; see [docs/setup/SETUP_INSTANCE.md §0](docs/setup/SETUP_INSTANCE.md#0-pre-flight-do-this-before-booting-the-instance) |
| Host / provider | Vast.ai (docker image), Verda (bare Ubuntu, no image), RunPod (docker, `/workspace` mount), in-house, ALICE HPC |
| Persistent mount path | `/workspace` on Vast/RunPod; `/root` on Verda; varies elsewhere |

Use [AskUserQuestion](#) (or plain prompting) to collect these before proceeding. If the user has already provided some fields in the same message, only ask for the missing ones.

### Step 2 — agent picks the right milestone + config + setup path

Decision table (refine with the actual answer):

| GPU shape | Recommended config | Launch script | Setup path |
|---|---|---|---|
| 1× B300 SXM6 (275 GB) | M5.5 `m5_5_research_paper_b300.yaml` (micro=4, act-ckpt off) | [training_m5_5/scripts/start_b300.sh](training_m5_5/scripts/start_b300.sh) (or [run.sh --mode prod_b300](training_m5_5/scripts/run.sh)) | [SETUP_INSTANCE.md §10](docs/setup/SETUP_INSTANCE.md#10-variant-verda-b300-fresh-ubuntu-no-docker-image) (Verda bare Ubuntu) |
| 2× B300 SXM6 (275 GB ×2) | M5.5 `m5_5_research_paper_b300_2xgpu.yaml` (TP=2, custom Qwen3.5 plan) | [start_b300.sh --mode prod_b300_2xgpu](training_m5_5/scripts/start_b300.sh) | same as 1× B300 — bootstrap is identical |
| 1× B200 (192 GB) | M5.1 `m5_1_research_paper.yaml` (micro=2, act-ckpt on) | [training_m5_1/scripts/run.sh](training_m5_1/scripts/run.sh) | [SETUP_INSTANCE.md §1–§9](docs/setup/SETUP_INSTANCE.md) (RunPod with `/workspace`) |
| 1× H200 (141 GB) | M5.5 `m5_5_research_paper_h200.yaml` (micro=2, gpu_mem=0.5, act-ckpt on) | [start_b300.sh --mode prod_h200](training_m5_5/scripts/start_b300.sh) | bootstrap is provider-agnostic; Spheron ES = `ubuntu@` not `root@`, use `sudo su -` after SSH |
| 1× H200 (141 GB) | M5.1 `m5_1_research_paper.yaml` (F1-only ablation) | [training_m5_1/scripts/run.sh](training_m5_1/scripts/run.sh) | [SETUP_INSTANCE.md §1–§9](docs/setup/SETUP_INSTANCE.md) |
| 1× H100 / A100 (80 GB) | M5.1 `m5_1_research_paper.yaml` | [training_m5_1/scripts/run.sh](training_m5_1/scripts/run.sh) | [SETUP_INSTANCE.md §1–§9](docs/setup/SETUP_INSTANCE.md) |
| 2× A100 (80 GB) | M5.5 `m5_5_research_paper_2xa100.yaml` (TP=2) | [training_m5_5/scripts/run.sh --mode prod_2xa100](training_m5_5/scripts/run.sh) | [SETUP_INSTANCE.md §1–§9](docs/setup/SETUP_INSTANCE.md) |
| 1× 24 GB (4090) | Eval only — too small for training | n/a | [SETUP_INSTANCE.md §1–§9](docs/setup/SETUP_INSTANCE.md), `SKIP_V2_BUILD=1` |
| Other / unlisted GPU | Ask user which milestone (M5.1 / M5.5 / M5.6) and start with the closest config; smoke-test first | `--mode smoke` on the chosen scaffold | [SETUP_INSTANCE.md](docs/setup/SETUP_INSTANCE.md) + [BOOTSTRAP_NEW_INSTANCE.md](docs/setup/BOOTSTRAP_NEW_INSTANCE.md) |

If the user has not picked a reward ablation, default to **M5.1 (F1-only)** since it's the production baseline.

### Step 3 — agent runs the chosen flow

1. Confirm provisioning is done (retriever assets, venvs, weights, .env with `WANDB_API_KEY`). The setup doc for the chosen path is the source of truth; bootstrap is idempotent.
2. Smoke first if the hardware shape is new: `bash <scaffold>/scripts/run.sh --mode smoke` (50 steps, ~5–10 min). Confirm the loop ends-to-ends before committing the multi-day prod run.
3. Launch prod under `tmux` (or `nohup … & disown`) so the SSH session can drop without killing training. The B300 launcher [training_m5_5/scripts/start_b300.sh](training_m5_5/scripts/start_b300.sh) is the reference for what a hardware-specific launcher looks like (pre-flight → retriever bring-up → tmux'd training).
4. Tail the log and surface the W&B run URL to the user.

**Special case — B300 / Verda one-shot path**: on a fresh Verda B300 box, one command does it all:

```bash
bash training_m5_5/scripts/start_b300.sh --smoke-first --mode prod_b300
```

That chains: bootstrap (if needed) → retriever bring-up → **smoke** (4 steps, ~5-10 min) → **prod** (auto-launches only if smoke reached Step 4/4). If smoke fails, prod does NOT launch and the chain exits with the failure. Both phases share one tmux session (`train`); detach with Ctrl-b d, reattach with `tmux attach -t train`.

For the manual one-step-at-a-time flow, `start_b300.sh` (no `--smoke-first`) auto-detects missing prerequisites and invokes [training_m5_5/scripts/bootstrap_b300.sh](training_m5_5/scripts/bootstrap_b300.sh). That script bakes in every fix from the 2026-05-15 bring-up (CUDA 12.9 swap, IB headers, ninja, cmake 4.x, cuDNN, uv system-wide symlink, V2 venv built from host shell with `NVTE_CUDA_ARCHS="90;120"`, Qwen3.5-0.8B HF pre-cache) and **prompts the user for `HF_TOKEN` if anonymous HF download fails**. The token is persisted to `training_m5_5/.env`. Lessons-learned + root-cause for each fix: [docs/setup/B300_RUNBOOK.md](docs/setup/B300_RUNBOOK.md).

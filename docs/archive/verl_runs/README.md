---
title: verl_runs archived training outputs
tags: [archive, verl]
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# verl_runs — Archived Training Run Outputs

Local-only archive of all 29 W&B runs analysed in the v0 + v1 reports. Pruned from the original 306 GB `/home/s4374886/verl/` (which also contained smoke-tests, abandoned configs, and multi-GPU experiments not in any report) down to ~238 GB on 2026-05-06.

Layout: one folder per report run, named `{report_name}_{wandb_id}`. Each folder contains:

- `wandb/run-{ts}-{wandb_id}/` — full W&B run record (config, history, files)
- `training.log` — per-run training log (or a copy of the shared log if the run was launched into a shared output dir)
- `global_step_{N}/` — actor + optimiser state at iteration N (only saved for runs that reached a checkpoint period)
- `rollout_{ts}/` — rollout JSONL traces from training (only for runs launched in their own output dir)
- `latest_checkpointed_iteration.txt` — pointer to the most recent global_step

All contents are gitignored (see `.gitignore`); only this README and `.gitignore` are tracked.

## Cross-reference: report name ↔ W&B id ↔ folder

### v0 — W&B project [`research`](https://wandb.ai/gaurisankarj1996-leiden-university/research) (14 runs)

| Report name | W&B id | Folder | Size | Has ckpts? |
|---|---|---|---:|---|
| `setup_first_pipeline` | `h3ga5d0w` | [`v0/setup_first_pipeline_h3ga5d0w/`](./v0/setup_first_pipeline_h3ga5d0w/) | <1 KB | no, see [`NOTE.md`](./v0/setup_first_pipeline_h3ga5d0w/NOTE.md) |
| `setup_run0_old_prompt` | `0wx183ke` | [`v0/setup_run0_old_prompt_0wx183ke/`](./v0/setup_run0_old_prompt_0wx183ke/) | 20 M | no |
| `setup_new_prompt_smoke` | `1oku1vc8` | [`v0/setup_new_prompt_smoke_1oku1vc8/`](./v0/setup_new_prompt_smoke_1oku1vc8/) | 3 M | no |
| `setup_stable_regime` | `ykxpxapv` | [`v0/setup_stable_regime_ykxpxapv/`](./v0/setup_stable_regime_ykxpxapv/) | 10 M | no |
| `setup_iter2_smoke` | `89yif4ob` | [`v0/setup_iter2_smoke_89yif4ob/`](./v0/setup_iter2_smoke_89yif4ob/) | 9 M | no |
| `p0_paper_w_ex` | `fj9ew2ik` | [`v0/p0_paper_w_ex_fj9ew2ik/`](./v0/p0_paper_w_ex_fj9ew2ik/) | 7.3 G | step_1000 |
| `p_minimal` | `un4quq94` | [`v0/p_minimal_un4quq94/`](./v0/p_minimal_un4quq94/) | 23 G | step_500/1000/1500 |
| `p1_basic_w_ex` | `z7kcxfof` | [`v0/p1_basic_w_ex_z7kcxfof/`](./v0/p1_basic_w_ex_z7kcxfof/) | 8.8 G | step_1000 (rollout snapshot + HF export) |
| `p1_basic_no_ex` | `e8l6r2kd` | [`v0/p1_basic_no_ex_e8l6r2kd/`](./v0/p1_basic_no_ex_e8l6r2kd/) | 34 M | no |
| `p2_basic2_w_ex` | `6dl2fz14` | [`v0/p2_basic2_w_ex_6dl2fz14/`](./v0/p2_basic2_w_ex_6dl2fz14/) | 47 M | no |
| `p2_basic2_no_ex` | `1cuveici` | [`v0/p2_basic2_no_ex_1cuveici/`](./v0/p2_basic2_no_ex_1cuveici/) | 33 M | no |
| `p3_decide_w_ex` | `0rjkbaa1` | [`v0/p3_decide_w_ex_0rjkbaa1/`](./v0/p3_decide_w_ex_0rjkbaa1/) | 23 G | step_500/1000/1500 |
| `p3_decide_no_ex` | `el6s2d2h` | [`v0/p3_decide_no_ex_el6s2d2h/`](./v0/p3_decide_no_ex_el6s2d2h/) | 30 G | step_500/1000/1500/2000 |
| `p4_think_w_ex` | `2jfi1l4c` | [`v0/p4_think_w_ex_2jfi1l4c/`](./v0/p4_think_w_ex_2jfi1l4c/) | 30 G | step_500/1000/1500/2000 |

### v1 — W&B project [`research_revamp`](https://wandb.ai/gaurisankarj1996-leiden-university/research_revamp) (15 runs)

| Report name | W&B id | Folder | Size | Has ckpts? |
|---|---|---|---:|---|
| `r0_strict_contract` | `xtcb7mo9` | [`v1/r0_strict_contract_xtcb7mo9/`](./v1/r0_strict_contract_xtcb7mo9/) | 36 M | no |
| `r1_query_object` | `0bhfwm68` | [`v1/r1_query_object_0bhfwm68/`](./v1/r1_query_object_0bhfwm68/) | 7.6 G | step_500 |
| `r2_concise` | `gzz5amvj` | [`v1/r2_concise_gzz5amvj/`](./v1/r2_concise_gzz5amvj/) | 15 G | step_500/1000 |
| `r3_80gb_a` | `x7ya3ev3` | [`v1/r3_80gb_a_x7ya3ev3/`](./v1/r3_80gb_a_x7ya3ev3/) | 2 M | no |
| `r3_80gb_b` | `jv6g32zu` | [`v1/r3_80gb_b_jv6g32zu/`](./v1/r3_80gb_b_jv6g32zu/) | 5 M | no |
| `r3_80gb_c` | `iz3w4cxj` | [`v1/r3_80gb_c_iz3w4cxj/`](./v1/r3_80gb_c_iz3w4cxj/) | 5 M | no |
| `exp_zero` | `cse8dhqk` | [`v1/exp_zero_cse8dhqk/`](./v1/exp_zero_cse8dhqk/) | 116 M | no |
| `exp_one` | `urlw74yz` | [`v1/exp_one_urlw74yz/`](./v1/exp_one_urlw74yz/) | 68 M | no |
| `base_state_machine_a` | `hmf76bfd` | [`v1/base_state_machine_a_hmf76bfd/`](./v1/base_state_machine_a_hmf76bfd/) | 30 G | step_500/1000/1500/2000 |
| `base_state_machine_b` | `zrphud77` | [`v1/base_state_machine_b_zrphud77/`](./v1/base_state_machine_b_zrphud77/) | 30 G | step_500/1000/1500/2000 |
| `base_with_example_a` | `guzkoeg4` | [`v1/base_with_example_a_guzkoeg4/`](./v1/base_with_example_a_guzkoeg4/) | 42 M | no |
| `base_with_example_b` | `d5ey6zj9` | [`v1/base_with_example_b_d5ey6zj9/`](./v1/base_with_example_b_d5ey6zj9/) | 46 M | no |
| `base_breakthrough` | `b8vv0qe2` | [`v1/base_breakthrough_b8vv0qe2/`](./v1/base_breakthrough_b8vv0qe2/) | 30 G | step_500/1000/1500/2000 |
| `think_prefill` | `huw425rb` | [`v1/think_prefill_huw425rb/`](./v1/think_prefill_huw425rb/) | 13 M | no |
| `reward_v3` | `w00kccwq` | [`v1/reward_v3_w00kccwq/`](./v1/reward_v3_w00kccwq/) | 7.6 G | step_500 |

## Notes on shared-output runs (v0)

Five v0 report runs share an output dir with multiple sibling runs because the launcher script reused fixed dir names (`qwen3_0.6b_instruct_grpo_gpu_1_80gb`, `qwen3_0.6b_instruct_grpo_gpu_1_40gb`) before the timestamp-suffix convention was adopted. For these, the shared `training.log` is **copied** into each per-run folder (so each folder is self-contained), and the surviving `global_step_*` checkpoints are attributed to the latest writer of that dir:

- `qwen3_0.6b_instruct_grpo_gpu_1_80gb` (3 runs: `setup_run0_old_prompt`, `setup_new_prompt_smoke`, `setup_stable_regime`) — never reached the checkpoint period, no `global_step_*` saved.
- `qwen3_0.6b_instruct_grpo_gpu_1_40gb` (6 runs: `setup_iter2_smoke`, `p0_paper_w_ex`, `p1_basic_w_ex`, `p1_basic_no_ex`, `p2_basic2_w_ex`, `p2_basic2_no_ex`):
  - `global_step_1000` is attributed to `p0_paper_w_ex` (`fj9ew2ik`) — the latest run to write to this dir on 2026-04-08.
  - `global_step_1000_rollout_20260407_002729` and its `_hf` export are attributed to `p1_basic_w_ex` (`z7kcxfof`) — the rollout timestamp matches `z7kcxfof`'s W&B start time (run-20260407_002925).

The v1 80gb shared dir (`r3_80gb_a/b/c`) has no checkpoints (all three crashed under 100 steps).

## Related

- [`docs/report/RESULTS_m0_a.md`](../../report/RESULTS_m0_a.md) — v0 results and per-run analysis (14 runs)
- [`docs/report/RESULTS_m0_b.md`](../../report/RESULTS_m0_b.md) — v1 results and per-run analysis (15 runs)
- [`docs/report/CODE_SETUP_m0.md`](../../report/CODE_SETUP_m0.md) — full codebase doc for the retired `re-search` repo

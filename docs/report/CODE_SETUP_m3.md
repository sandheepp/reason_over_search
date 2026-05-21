---
title: Code Setup M3 — Qwen3-0.6B evaluation pipeline (M3 + M3.1)
tags: [report, eval, m3, m3.1]
source: internal
created: 2026-05-07
updated: 2026-05-08
---

# Code Setup M3: Qwen3-0.6B Evaluation Pipeline (M3 + M3.1)

**Date**: 2026-05-07 (M3); extended 2026-05-08 (M3.1, see §13.5).  
**Scope**: What changed from the Search-R1-style eval port (`evaluation_search_r1/`, used for the M1 paper-baseline reproduction) to the M3 evaluation pipeline (`evaluation_research/`) that aligns the eval to the **`p1_basic_w_ex_z7kcxfof`** training rollout (Phase-1 v0 block, the only run that converged on heavy-tool 2-call / 4-turn behaviour). M3.1 extends the same pipeline to the highest-reward Phase-1 run, `p3_decide_no_ex_el6s2d2h`.  
**Cluster (training, original)**: ALICE 1× A100-40GB (Phase-1 GRPO, 2026-04-06 / 2026-04-09).  
**Cluster (M3 eval)**: ALICE 1× A100-80GB (`gpu-short` + `gpu-a100-80g`).  
**Source path in this repo**: `evaluation_research/` (overlay copy of `evaluation_search_r1/` with M3 code changes), `local_retriever/`, `scripts/run_m3.sh`, `scripts/sbatch_m3.sh`.

---

## 1. Headline Diff vs the Search-R1 Eval Port (`evaluation_search_r1/`)

| Dimension | Search-R1 eval port (M1) | M3 eval (this doc) |
|---|---|---|
| **Model** | `Qwen2.5-3B-Search-R1` (paper checkpoints, base + instruct) | **`Qwen3-0.6B`** hybrid (pre-GRPO) and **`qwen_3_0.6b_v0`** (post-1046-step GRPO with `p1_basic_w_ex` prompt) |
| **Prompt arm** | Search-R1 `SEARCH_R1_TEMPLATE` (user-message format) | **`QWEN3_0_6B_TEMPLATE`** (system-message verbatim from `p1_basic_w_ex` training prompt) |
| **Action tags** | `<search>` / `<information>` | `<search>` / `<result>` (matches training; `<information>` was the Search-R1 paper choice) |
| **Retrieval text format** | `"Doc i (Title: <title>) <text>\n"` joined (Search-R1 `_passages2string`) | **Raw `{contents}\n\n` joined and stripped** (matches verl-legacy training rollout `vllm_rollout.py:286-290`) |
| **Result wrapper into next turn** | `\n\n<information>{X}</information>\n\n` | **`" <result>\n{X}\n</result>"`** with **leading space** (byte-identical to verl-legacy `vllm_rollout.py:419` `tokenizer.encode(f" <result>\n{result}\n</result>")`) |
| **Top-k retrievals** | top-3 (paper-fidelity) | **top-5** (matches training `top_n=5` from `vllm_rollout.py:286`) |
| **Max obs length** | 500 tokens (Search-R1 evaluate.sh default) | **256 tokens** (matches training `max_tool_response_length=256` from `training.log:289`) |
| **Total response budget** | 4× 500 = 2000 tokens (4 turns × per-step cap) | **4096 tokens** total (matches training `response_length=4096`); **no per-step cap** — `min(remain_length, step_limit)` reduces to `remain_length` |
| **Max search turns** | 4 | **5** (matches training observed `num_turns/max:5`) |
| **`enable_thinking`** | `False` (instruct/base do not have hybrid thinking) | **`True`** — Qwen3 hybrid chat template otherwise auto-injects `<think>\n\n</think>\n\n` to suppress thinking; with `True` the model decides |
| **Pred extraction** | Content of `<answer>...</answer>` (regex) | Content of `<answer>...</answer>`, then **unwrap `\boxed{X}`** to return `X` (qwen3 mode); accepts `\boxed{{X}}` defensively |
| **Retriever index** | Flat IP × 2 workers (paper-fidelity) | **IVF-SQ8 × 8 workers** (matches training-time v1 default; ~10× faster, recall hit < 1 %) |
| **Retriever paths** | `local_retriever/{corpus,indexes,models}/` (subdir-relative) | **Project-root** `corpus/`, `indexes/`, `models/` (so `retriever_config.yaml` relative paths resolve when launched from `$REPO_ROOT`) |
| **Eval pipeline source** | `evaluation_search_r1/` | `evaluation_research/` (editable install in the same `evaluation_search_r1` conda env via `pip install -e evaluation_research/ --no-deps`) |
| **Hardware** | Vast.ai RTX 4090 24GB | ALICE A100 80GB (`gpu-short` + `gpu-a100-80g`) |

---

## 2. What's Unchanged (paper-faithful in both M1 and M3)

| Knob | Value | Notes |
|---|---|---|
| Decoding | `temperature=0.0` (greedy) | Single seed × Plan A; multi-seed redundant under greedy |
| `apply_chat=True` | both variants | Qwen3 hybrid uses `tokenizer.apply_chat_template` with system+user roles |
| Retriever encoder | `e5-base-v2` | unchanged from training; bf16 fp16 |
| Retriever HTTP contract | `POST /search` on `127.0.0.1:3005` | identical to training rollout |
| SGLang server contract | `POST /generate` + `GET /health` on `127.0.0.1:3000` | unchanged |
| Eval datasets | bamboogle, nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique | 7 datasets, full Plan A test/dev sets — see §10 |
| Metrics | EM, ACC, F1 (FlashRAG `evaluator/metrics.py`) | unchanged |

---

## 3. Critical Bugs Fixed During M3 Smoke (14 fixes between clone and the first clean GRPO-vs-base comparison)

Each item below blocked the eval at one stage. All were caught and fixed during the slot 1 (NODE_FAIL'd) and slot 2 (interactive) sessions on 2026-05-06/07. The fixes are all in `evaluation_research/`, `scripts/sbatch_m3.sh`, `scripts/run_m3.sh`, and the project-root `corpus/`/`indexes/`/`models/` symlinks.

| # | File | Symptom | Fix |
|---|---|---|---|
| 1 | `local_retriever/retriever_config.yaml` (relative paths) | `FileNotFoundError: '/zfsstore/.../reason_over_search/./corpus/wiki18_100w.jsonl'` when retriever launched from project root with only `--index` overridden; corpus_path stayed relative | **Moved corpus / index / models to project root** (`corpus/wiki18_100w.jsonl` symlink, 16 GB `indexes/wiki18_100w_e5_ivf4096_sq8.index` mv-rename, `models/e5-base-v2/` symlink). `local_retriever/{corpus,indexes,models}/` removed. Relative paths in the config now resolve from `cd $REPO_ROOT`. |
| 2 | `evaluation_search_r1` conda env | `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'` — flashinfer JIT compile fails | **`pip install ninja`** in the env (one-time, persists in `/home/.../envs/evaluation_search_r1/`). |
| 3 | `sbatch_m3.sh` / interactive ssh launch | `RuntimeError: Could not find CUDA installation. Please set CUDA_HOME environment variable` — flashinfer / tvm_ffi need nvcc | **Source lmod init**: `source /etc/profile.d/lmod.sh && module load CUDA/12.4.0` before launching SGLang. Sets `CUDA_HOME=/easybuild/software/CUDA/12.4.0`. In `sbatch` jobs lmod runs automatically (the script's `module load CUDA/12.4.0` works). In `ssh` sessions from outside an `srun --pty` shell, lmod does not auto-init — must source `/etc/profile.d/lmod.sh` explicitly. |
| 4 | `evaluation_research/flashrag/search_r1/templates.py` line 8 | Model output `\boxed{{Yes}}` (double braces) instead of `\boxed{Yes}` — every EM = 0 | Template had `\\boxed{{answer here}}` (Python `{{}}` escape) but is used as a **raw** string for qwen3 mode (no `.format()` call). Changed to `\\boxed{answer here}` (single braces). Matches training prompt verbatim. |
| 5 | `evaluation_research/flashrag/search_r1/reward.py:82` (`extract_solution`) | `pred = "The final answer is \[ \boxed{Yes} \]"` instead of `pred = "Yes"` | After extracting `<answer>...</answer>` content, **unwrap `\boxed{X}`** with regex `r"\\boxed\{+\s*(.+?)\s*\}+"` (tolerates `\boxed{X}` and defensively `\boxed{{X}}`). Returns inner X for EM/F1 comparison. |
| 6 | `evaluation_research/flashrag/pipeline/active_pipeline.py` (qwen3 mode result wrapper) | `query += f"{output_str}<result>\n{retrieval_text}\n</result>"` — **no leading space** before `<result>`; tokenization differs from training | Changed to `query += f"{output_str} <result>\n{retrieval_text}\n</result>"` — **leading space** matches `verl_legacy/vllm_rollout.py:419` `tokenizer.encode(f" <result>\n{result}\n</result>")`. Observation tokens are loss-masked during training (line 421: `result_mask=[0]*len`), so the model has only seen this exact framing in context. |
| 7 | `evaluation_research/flashrag/pipeline/active_pipeline.py` (retrieval text format) | `retrieval_text` formatted as `"Doc 1 (Title: X) Y\nDoc 2 (Title: Z) W\n"` (Search-R1 `_passages2string`); training fed raw `{contents}\n\n` joined | Branched on `prompt_mode`: when `'qwen3'`, build retrieval_text as `''.join(f"{line['contents']}\n\n" for line in search_result).strip()` (matches `vllm_rollout.py:286-290`). When `'search_r1'`, keep the Doc-i format (paper-fidelity). |
| 8 | `evaluation_research/run_eval.py` (in `search_r1()` function) | `retrieval_topk: 3` from `basic_config.yaml` for both arms; training used `top_n=5` (`vllm_rollout.py:286 data = {'query':..., 'top_n':5}`) | Override per-mode: `if args.prompt_mode == "qwen3": config_dict["retrieval_topk"] = 5`. `search_r1` mode keeps top-3 (paper-fidelity). |
| 9 | `evaluation_research/flashrag/pipeline/active_pipeline.py:64-75` (per-mode budgets) | Hardcoded `step_limit=500`, `max_obs_length=500`, `max_search_turns=4`; training had `response_length=4096`, `max_tool_response_length=256`, observed up to 5 turns | Branched on `prompt_mode`: when `'qwen3'`, set `max_search_turns=5`, `step_limit=8192` (effectively no per-step cap; bounded by `remain_length`), `max_obs_length=256`. When `'search_r1'`, keep paper defaults (4/500/500). |
| 10 | `evaluation_research/flashrag/config/basic_config.yaml:105` | `generator_max_input_len: 4096` was actually correct for qwen3, but originally bumped to 8192 mid-debug then reverted | Final value `generator_max_input_len: 4096` matches training `response_length=4096`. SGLang `--context-length 8192` provides headroom; the 4096 cap matches training distribution. |
| 11 | `scripts/run_m3.sh` (CLI defaults to `enable_thinking=False`) | Qwen3 chat template injects `<think>\n\n</think>\n\n` (closed empty think block) before the model generates anything; the model has no open block and emits short, unhelpful answers | Pass `--enable_thinking True` from `run_m3.sh` so `tokenizer.apply_chat_template(..., enable_thinking=True)` lets the model decide whether to emit `<think>` (training rollouts show the model does emit `<think>...</think>` before each `<search>`; verified in `re-search/examples/rollout_20260407_104124_top5_whole_datapoint.jsonl:1`). |
| 12 | `scripts/sbatch_m3.sh` `REPO_ROOT` resolution | `REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"` — under `sbatch`, SLURM copies the script to `/var/spool/slurmd/<jobid>/slurm_script`, so `dirname/..` resolves to `/var/spool/slurmd/`. Result: `IVF index not found` (wrong path), then `python: can't open file '/var/spool/slurmd/local_retriever/retriever_serving.py'`. Job 2121202 failed in 9 s. | Use `REPO_ROOT="${SLURM_SUBMIT_DIR:-/zfsstore/user/s4374886/omega/reason_over_search}"` — `SLURM_SUBMIT_DIR` is set by SLURM to the directory the user ran `sbatch` from; hardcoded fallback for direct execution. |
| 13 | `scripts/sbatch_m3.sh` retriever timeout | First-time HuggingFace `datasets` arrow conversion of `wiki18_100w.jsonl` (~21 M passages) takes 5–10 min on a cold node; the original 120 s timeout (`for i in $(seq 1 24); do sleep 5`) tripped before the retriever bound to port 3005. Job 2125008 failed in 2m05s. | Extended retriever timeout to 600 s (`for i in $(seq 1 120)`); SGLang timeout to 300 s. Capture retriever stdout/stderr to `logs/m3_<jobid>_retriever.log` and SGLang to `logs/m3_<jobid>_sglang.log` for post-mortem when health fails. Print `RETRIEVER_PID` and `SGLANG_PID` to `.out` for debugging. |
| 14 | (one-time) HuggingFace `datasets` cache key change after corpus path move | Moving `local_retriever/corpus/wiki18_100w.jsonl` → `corpus/wiki18_100w.jsonl` changes the HF cache config_id (`default-XXXXXX`) for that file path; the previously-built 13 GB arrow cache (`default-6d73c569a8299e2e/0.0.0/fd36e7...`) was still on disk under the OLD key. First retriever launch on the new path would have rebuilt the cache (~5 min). | **Hard-link the existing complete cache** to the new key location: `cp -al .../default-6d73c569a8299e2e .../default-4681c96eb42e7473`. Same data hash `fd36e7...`, different config-id directory; HF datasets sees the complete arrow set under the new key, skips rebuild. Zero disk used. Subsequent launches on `node870` started workers in seconds. |

The 14 fixes plus an `setsid + nohup + PATH=` wrapper convention for ssh-detached background processes (next subsection) constitute the full delta from "clone + run" to "first clean comparison".

### `setsid + nohup + PATH=` wrapper (for ssh-driven launches, not sbatch)

When detaching long-lived processes from an ssh session, both `nohup` and `setsid` are needed — `nohup` alone keeps the child in the ssh session's process group and it dies on disconnect:

```bash
ssh <node> "cd $PROJ && setsid bash -c 'PATH=/path/to/env/bin:\$PATH nohup ... > log 2>&1 < /dev/null & echo \$! > pid'"
```

Without explicit `PATH=…`, the conda env's binaries (e.g. `ninja`) are not found by subprocess JIT compilers (flashinfer, tvm_ffi).

`sbatch` jobs do not need this — the SLURM batch shell is its own session and runs even if the user disconnects.

---

## 4. Architecture of `evaluation_research/` (overlay copy of `evaluation_search_r1/`)

`evaluation_research/` is a full copy of `evaluation_search_r1/` (which itself is a port of FlashRAG's eval pipeline to support Search-R1) with three targeted edits. Editable install of `evaluation_research/` in the `evaluation_search_r1` conda env (via `pip install -e evaluation_research/ --no-deps`) overrides the `flashrag` package import, so `import flashrag` resolves to `evaluation_research/flashrag/`.

```
evaluation_research/
  flashrag/
    search_r1/
      templates.py         — QWEN3_0_6B_TEMPLATE (verbatim p1_basic_w_ex prompt) added before SEARCH_R1_TEMPLATE
      reward.py            — extract_solution() unwraps \boxed{X} from <answer>...</answer> content (qwen3 mode)
    pipeline/
      active_pipeline.py   — SearchR1Pipeline.__init__(prompt_mode='search_r1' | 'qwen3'); per-mode budgets, retrieval format, result wrapper
      parallelism.py       — INFERENCE_MAX_WORKERS = 32 (bumped from 16 for 2× speedup on warm SGLang)
    config/
      basic_config.yaml    — generator_max_input_len: 4096; retrieval_topk: 3 (qwen3 overridden to 5 in run_eval.py)
  run_eval.py              — argparse adds --prompt_mode {search_r1, qwen3}; sets retrieval_topk=5 when qwen3
  results/                 — per-(dataset, save_note) intermediate_data.json + metric_score.txt + config.yaml
```

The four files touched (`templates.py`, `reward.py`, `active_pipeline.py`, `parallelism.py`, `run_eval.py`, `basic_config.yaml`) are the entire M3 surface area. Everything else is unchanged from the upstream FlashRAG / Search-R1 port.

---

## 5. Prompt — `p1_basic_w_ex` verbatim (system message)

The `QWEN3_0_6B_TEMPLATE` in `evaluation_research/flashrag/search_r1/templates.py` is the byte-for-byte system message used in the `p1_basic_w_ex_z7kcxfof` training run (verified against `docs/report/RESULTS_m0_a.md:175-189`):

```text
You are a helpful assistant who can answer questions using multiple Wikipedia search tool calls.
You can call the search tool by writing: <search> your query </search>
You will receive the result in: <result> your search result </result>
Use the search tool to obtain the information needed for the answer.
Answers should be based on the search results.
You may use the search tool multiple times if needed before giving the final answer.
Provide the final answer in the format: <answer>The final answer is \[ \boxed{answer here} \]</answer>.
For example:
Question: What is the nationality of the author of Hamlet?
<search>Hamlet</search>
<result>The Tragedy of Hamlet was written by William Shakespeare.</result>
<search>William Shakespeare</search>
<result>William Shakespeare was an English playwright.</result>
<answer>The final answer is \[ \boxed{English} \]</answer>
```

Rendered into the chat as: `[{role: 'system', content: <above>}, {role: 'user', content: <question>}]` then `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`. The user message is the literal benchmark question (no template substitution).

---

## 6. Per-mode dispatch (`prompt_mode`)

`SearchR1Pipeline.__init__(prompt_mode='search_r1' | 'qwen3')` is the single switch that swaps eight things atomically:

| Setting | `'search_r1'` (M1 paper-fidelity) | `'qwen3'` (M3, this work) |
|---|---|---|
| Template | `SEARCH_R1_TEMPLATE` (user-message, `{prompt}` substituted via `.format`) | `QWEN3_0_6B_TEMPLATE` (system-message, used as raw string) |
| Chat construction | `[{role: 'user', content: SEARCH_R1_TEMPLATE.format(prompt=q)}]` | `[{role: 'system', content: QWEN3_0_6B_TEMPLATE}, {role: 'user', content: q}]` |
| Action stop tokens | `</search>`, `</answer>`, `<\|im_end\|>`, `<\|endoftext\|>` | same |
| Retrieval text | Doc-i format (`Doc 1 (Title: X) Y\n…`) | raw `{contents}\n\n` joined, stripped |
| Result wrapper | `\n\n<information>{X}</information>\n\n` | `" <result>\n{X}\n</result>"` (leading space) |
| `retrieval_topk` | 3 (paper) | 5 (training) |
| `max_obs_length` | 500 tokens | 256 tokens |
| `max_search_turns` | 4 | 5 |
| `step_limit` | 500 | 8192 (no per-step cap; bounded by remain_length) |

---

## 7. Reward / metrics (FlashRAG evaluator)

Same evaluator as M1: `flashrag/evaluator/metrics.py` returns three numbers per item, averaged across the dataset:

- **EM** (`exact_match_score`): 1.0 if `normalize_answer(pred) == normalize_answer(any_golden)` else 0.0. `normalize_answer` lowercases, strips articles (a/an/the), removes punctuation, normalises whitespace.
- **ACC** (`acc_score`): 1.0 if `normalize_answer(any_golden) in normalize_answer(pred)` else 0.0 (substring containment).
- **F1** (`f1_score`): token-overlap F1 on `normalize_answer`'d strings.

Pred extraction (`extract_solution` in `evaluation_research/flashrag/search_r1/reward.py`):
```python
def extract_solution(solution_str: str):
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches: return None
    answer_text = matches[-1].group(1).strip()
    # qwen3: unwrap \boxed{X} (also tolerates accidental \boxed{{X}})
    boxed = re.search(r"\\boxed\{+\s*(.+?)\s*\}+", answer_text, re.DOTALL)
    return boxed.group(1).strip() if boxed else answer_text
```

The `\boxed{+ … }+` pattern tolerates the (now-fixed) double-brace bug from §3 #4 — defensive in case some old result file is re-evaluated.

---

## 8. Key Qwen3-0.6B Hybrid Constraints Surfaced During Eval

1. **`enable_thinking=True` is required** (same finding as M2 NeMo-RL training, `CODE_SETUP_m2.md` §7.2). With `False`, Qwen3 hybrid's chat-template Jinja inserts `<think>\n\n</think>\n\n` before the assistant turn — a *closed* empty think block. The model has no open block to fill and emits short shallow answers. With `True`, no auto-injection; the training rollouts show the post-GRPO model emits `<think>...</think>` before each `<search>` (verified against `re-search/examples/rollout_20260407_104124_top5_whole_datapoint.jsonl:1`).

2. **flashinfer attention backend is the right choice on A100 + CUDA 12.4** (mid-debug we briefly tried `--attention-backend triton --disable-cuda-graph` to bypass missing nvcc; once `module load CUDA/12.4.0` was sourced, default flashinfer with cuda-graph for decode worked). SGLang's per-batch log shows `cuda graph: True` for `Decode batch` lines (the `cuda graph: False` on `Prefill batch` lines is normal — prefill never uses cuda graphs).

3. **SGLang context length must be ≥ training response_length**. We launched with `--context-length 8192` and capped `generator_max_input_len: 4096` in the eval config; the cap matches training distribution while leaving SGLang headroom.

---

## 9. Hardware sizing (IVF-SQ8 × 8 workers)

Same numbers as M2 retriever sizing. Retriever runs the `wiki18_100w_e5_ivf4096_sq8.index` (16 GB) with 8 parallel workers (`--num_retriever 8`), `OMP_NUM_THREADS=4` per worker.

| Component | Count | Sizing |
|---|---|---|
| FAISS OMP threads | 8 workers × 4 = **32** threads | Diminishing returns beyond 4/worker at nprobe=64 |
| E5-base-v2 encoder | 8 concurrent | Single-threaded per call; absorbed in OMP budget |
| FastAPI / uvicorn | ~4 | Event-loop + worker threads |
| SGLang (tokeniser + scheduler) | ~4 | |
| **Total CPU** | **~40 cores** | `--cpus-per-task 40` |
| 8 × IVF-SQ8 index in RAM | 8 × 16 GB ≈ **128 GB** | Measured 134 GB RSS at peak |
| 8 × E5-base-v2 encoder | 8 × 440 MB ≈ 3.5 GB | |
| Qwen3-0.6B SGLang (bf16) | ~1.5 GB VRAM | Negligible host-RAM |
| Corpus jsonl + Python overhead | ~10 GB | |
| **Total RAM** | **~150 GB** | `--mem=160g` (A100 nodes have ≥ 256 GB host) |
| GPU VRAM | 1× A100 80GB | Heavily under-utilised but required for the partition |

`OMP_NUM_THREADS=4` must be set **before** launching the retriever to keep FAISS within the per-worker budget.

---

## 10. Plan A (full eval sets, not Plan B subsamples)

`sample_num` is **not** respected by the `search_r1` pipeline path — both pre-GRPO and v0 ran on the **full test/dev sets** rather than the 1k stratified subsamples MILESTONE_3.md originally planned (Plan B). This is a happy upgrade: the comparison is statistically more rigorous, with no subsampling noise.

| Dataset | Split | Items (raw `data/{ds}/{split}.jsonl` line count = pre-GRPO result count = v0 result count) |
|---|---|---:|
| nq | test | 3,610 |
| triviaqa | test | 11,313 |
| popqa | test | 14,267 |
| hotpotqa | dev | 7,405 |
| 2wikimultihopqa | dev | 12,576 |
| musique | dev | 2,417 |
| bamboogle | test | 125 |
| **Total per variant** | | **51,713** |

Both variants saw the identical 51,713 items → 103,426 total evaluations.

---

## 11. Wall-clock observed (A100 80GB, ALICE)

Two regimes recorded — `INFERENCE_MAX_WORKERS=16` (initial, pre-GRPO partial run) and `=32` (bumped mid-pre-GRPO and used for all v0).

| Dataset | Items | pre-GRPO 16w | pre-GRPO 32w | v0 32w |
|---|---:|---|---|---|
| bamboogle | 125 | smoke 28 s | smoke 28 s | smoke 29 s |
| nq | 3,610 | 7m25s | — | ~14m |
| triviaqa | 11,313 | 25m37s | — | ~31m |
| popqa | 14,267 | 29m52s | — | ~21m |
| hotpotqa | 7,405 | — | **15m24s** | ~21m |
| 2wikimultihopqa | 12,576 | — | 29m48s | ~16m |
| musique | 2,417 | — | 5m54s | ~8m |
| **Variant total** | 51,713 | (mixed) ~115 min | — | **~146 min** |

- 32-worker speedup vs 16-worker: hotpotqa 1000-item-equiv 15m24s vs popqa 1000-item-equiv 29m52s → **~2× faster** at the same per-dataset shape (per-item time 0.92 s vs 1.79 s).
- v0 wall-clock per dataset is comparable to pre-GRPO 32-worker; v0 emits more tokens per question (more `<think>` reasoning, often 2 search turns) so the 2× concurrency lift partially cancels.
- SGLang `gen throughput` peaks at ~3,300 tokens/s decode on the 0.6B model with cuda-graph + flashinfer.
- Cold start: retriever IVF + 8-worker arrow build = 5–10 min (first time only; HF cache shared via NFS persists across compute nodes).

---

## 12. Comparison to ReSearch paper setup

| Knob | ReSearch paper (`re_search/src/flashrag/pipeline/active_pipeline.py:18-181`, `ReSearchPipeline`) | M3 (this work) |
|---|---|---|
| Action format | `<tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>` (Hermes / ChatML JSON) | `<search>X</search>` (verbatim Search-R1 tags, used by `p1_basic_w_ex` training prompt) |
| Result wrapper | ` <tool_response>\n{X}\n</tool_response>` (leading space) | ` <result>\n{X}\n</result>` (leading space) |
| Retrieval text | `{contents}\n\n` joined, stripped | same |
| `top_n` | 5 | 5 |
| `max_search_turns` | 8 (ReSearch ceiling) | 5 (training observed max) |
| `step_limit` per call | 512 | 8192 (no cap; bounded by remain_length=4096) |
| `generator_max_input_len` | 1024 (ReSearch eval default in `basic_config.yaml`; smaller than training response_length) | 4096 (matches training response_length=4096) |
| Pred extraction | `last_boxed_only_string + remove_boxed` (LaTeX-style boxed extraction) | `<answer>...</answer>` then `\boxed{X}` regex unwrap (effectively the same) |
| `enable_thinking` | False default in re-search code | True (Qwen3 hybrid soft-switch needs it for the model to actually reason) |
| Prompt template variant | `re_search_template_sys` (system-prompt that emphasises `<think>`-then-act loop) | `QWEN3_0_6B_TEMPLATE` aka `p1_basic_w_ex` (basic rules + Hamlet 2-search example, no explicit `<think>` instruction; model emits `<think>` anyway because it was trained on Qwen3 hybrid which does this by default) |
| Reward at training time | EM-only, paper-faithful (verl_legacy `qa_em.py`) | same |
| Tags ablated | `<search>/<result>` (matches both the published Search-R1 paper and the published ReSearch paper — verified upstream at `re-search` commit `51d98e1`) **vs** `<tool_call>/<tool_response>` (Qwen-native; *our local ablation* introduced in `re-search` commit `2c32dd3`, 2026-04-12, and tested in our v1 block — **not** the published paper's scheme) | `<search>/<result>` only (we evaluate the v0 checkpoint trained with these tags) |

**Net**: M3 eval is byte-equivalent to a re-execution of the verl-legacy training rollout for `p1_basic_w_ex`, with the only deltas being (a) the SGLang inference backend instead of vLLM, (b) the eval-side wiring through FlashRAG, (c) the retriever served by `local_retriever/` instead of an in-process call.

---

## 13. Smoke results / progression of fixes (bamboogle, 125 items, smoke loop)

| Iteration | Fix introduced | EM | F1 | Note |
|---|---|---:|---:|---|
| 1 | Initial copy of `evaluation_search_r1/` with `prompt_mode=qwen3` wired but no template fix | 0.000 | 0.017 | All preds were `"The final answer is \[ \boxed{{X}} \]"` (double braces in prompt → double in output → predictor unwrap fails) |
| 2 | + Single-brace template + `extract_solution` unwraps `\boxed{}` | 0.008 | 0.055 | Only 1/125 EM; format works but model outputs are wrong (small model on multi-hop) |
| 3 | + Leading space before `<result>` (matches training tokenization) | n/a | n/a | (re-tested as part of 4) |
| 4 | + Raw retrieval format (no Doc-i) + top-5 + `enable_thinking=True` (full alignment) | 0.040 (pre-GRPO) / 0.080 (v0) | 0.080 / 0.140 | First time the lift between variants is visible |
| 5 | + Per-mode budgets (256 obs / 4096 total / 5 turns) + no per-step cap | **0.056 (pre-GRPO)** / **0.088 (v0)** | 0.085 / 0.140 | Final M3 config; matches user's expected ~0.07 / ~0.08 memory for bamboogle |

The +0.032 EM lift on bamboogle survives end-to-end and matches the user's prior empirical result for this checkpoint pair.

---

## 13.5. M3.1 extension (2026-05-08): second Phase-1 checkpoint, second prompt

After M3 closed, we evaluate `p3_decide_no_ex_el6s2d2h` — the Phase-1 v0 run with the **highest end-of-run rollout reward (0.215)** but a **different prompt** than z7kcxfof. This required no new alignment fixes; the M3 14-fix audit holds. Two purely additive changes to the pipeline:

1. **New prompt template constant**: `P3_DECIDE_NO_EX_TEMPLATE` in `evaluation_research/flashrag/search_r1/templates.py`, byte-for-byte identical to the system message used at training time (recovered from [`docs/report/RESULTS_m0_a.md`](RESULTS_m0_a.md) §`p3_decide_no_ex (el6s2d2h)`). Difference vs `QWEN3_0_6B_TEMPLATE` (= `p1_basic_w_ex`):
   - "Use the search tool **multiple** Wikipedia search tool calls" → "Use the search tool" (the "multiple" was the heavy-tool anchor in M3)
   - "Answers should be based on the search results" → "Use the information in the search results to determine the final answer" + two extra decision-rule sentences
   - **No Hamlet 2-search example** at the bottom (the headline structural difference)
2. **`prompt_mode='qwen3_p3_decide_no_ex'`** added as a new mode. Implementation: `templates.py` exports `QWEN3_TEMPLATES = {"qwen3": ..., "qwen3_p1_basic_w_ex": ..., "qwen3_p3_decide_no_ex": ...}`. `active_pipeline.py` and `run_eval.py` switched their `prompt_mode == 'qwen3'` checks to `prompt_mode.startswith('qwen3')` so all qwen3 modes share retrieval format / budgets / `enable_thinking=True`; only the system message differs.

**Checkpoint conversion**: el6s2d2h was archived in verl-FSDP format (`actor/model_world_size_1_rank_0.pt`) on the user's training machine — **the verl-FSDP run archives are not retained in this repo** (only the HF-converted safetensors live under `eval/`); converted to HF safetensors via:

```bash
# <ARCHIVE_DIR> is the verl-FSDP run output, e.g.
# /path/to/verl_runs/v0/p3_decide_no_ex_el6s2d2h/global_step_2000/actor
/home/s4374886/.conda/envs/verl-latest/bin/python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir <ARCHIVE_DIR> \
    --target_dir eval/qwen_3_0.6b_v0_no_ex
```

(~1 min on the login node; output is a 1.5 GB `model.safetensors` + tokenizer files. Step 2000 is the closest checkpointed step to end-of-run 2280; same approach as M3, which used z7kcxfof's step 1000 of 1046.)

**HF publication**: the converted checkpoint is also published to HuggingFace Hub at [`pantomiman/Qwen3-0.6B-v0.1`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0.1) (parallel to the M3 checkpoint at [`pantomiman/Qwen3-0.6B-v0`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0); the `.1` minor-version suffix distinguishes the no-example variant from the same base + algorithm + reward + data, varying only the prompt). To reproduce the upload:

```bash
export HF_TOKEN='<your_write_token>'
hf repo create pantomiman/Qwen3-0.6B-v0.1 --repo-type model
cd eval/qwen_3_0.6b_v0_no_ex
hf upload pantomiman/Qwen3-0.6B-v0.1 . . --repo-type model
```

The repo also ships a model card (`README.md` in the same directory) that documents the run id, training-time signature, action format (`<search>` / `<information>`, matching the published Search-R1 / ReSearch papers), and the `v0` ↔ `v0.1` relationship.

**Variant dispatch**: `scripts/run_m3.sh` and `scripts/sbatch_m3.sh` add a `qwen3_0.6b_v0_no_ex` variant case that points at `eval/qwen_3_0.6b_v0_no_ex/` and passes `--prompt_mode qwen3_p3_decide_no_ex`. Same SGLang flags, same retriever, same datasets.

**Readiness budgets — bumped on both `/health` waits (cold-cache cliffs).** Two consecutive M3.1 sbatches failed at the post-launch `/health`-readiness loops, both because the M3 reference timings were near the budget and a fresh node crossed the cliff:

| sbatch | Node | Failure | Fix in `scripts/sbatch_m3.sh` |
|---|---|---|---|
| 2134645 | node875 | SGLang `/health` timed out at 300 s (M3 ref 260 s) | bumped 300 → 600 s (`for i in $(seq 1 120); do sleep 5; …`) |
| 2134663 | node873 | Retriever `/health` timed out at 600 s (M3 ref 570 s) | bumped 600 → 1200 s (`for i in $(seq 1 240); do sleep 5; …`) |
| **2150167** | **node870** | — | **completed all 7 datasets in 1 h 32 m** |

In all three cases the retriever + checkpoint + pipeline were healthy at failure time; only the wait windows were too tight for cold-cache imports. Both fixes are pure-margin (the loops break the moment `/health` returns 200, so warm-cache wall-clock is unaffected). The two budgets are now symmetric at 1200 s for retriever / 600 s for SGLang.

For results see [`RESULTS_m3.md`](RESULTS_m3.md) §14 (sbatch job 2150167 after the two prior timeouts).

---

## 14. Pointers

- Training run that produced `qwen_3_0.6b_v0`: `docs/report/RESULTS_m0_a.md` §`p1_basic_w_ex (z7kcxfof)`
- M3 milestone narrative: `docs/milestone_3/MILESTONE_3.md`
- M3 evaluation results table: `docs/report/RESULTS_m3.md`
- Phase-1 ALICE training synthesis (29 runs): `docs/report/RESULTS_m0_a.md` (14) + `docs/report/RESULTS_m0_b.md` (15)
- M2 NeMo-RL training pipeline (Phase-2 forward): `docs/report/CODE_SETUP_m2.md`
- Original Search-R1 eval port (M1 paper-fidelity baseline): `evaluation_search_r1/` (preserved unchanged)
- ReSearch reference implementation cross-checked against: `/home/s4374886/omega/re-search/src/flashrag/pipeline/active_pipeline.py` (ReSearchPipeline) and `/home/s4374886/omega/re-search/src/verl_legacy/workers/rollout/vllm_rollout/vllm_rollout.py` (the verl_legacy rollout that produced the v0 checkpoint)

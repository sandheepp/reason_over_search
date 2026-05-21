---
title: Code Setup M4 — Qwen3.5-0.8B baseline evaluation pipeline (M4 + M4.1)
tags: [report, eval, m4, m4.1]
source: internal
created: 2026-05-08
updated: 2026-05-08
---

# Code Setup M4: Qwen3.5-0.8B Evaluation Pipeline (M4 + M4.1)

**Date**: 2026-05-08 (M4 + M4.1 prompt redesign same-day)
**Scope**: Documents what changed from the M3 evaluation pipeline ([`evaluation_research/`](../../evaluation_research/), Qwen3-0.6B with Search-R1 invented `<search>` / `<result>` tags, audited in [`CODE_SETUP_m3.md`](CODE_SETUP_m3.md)) to the M4 / M4.1 pipeline ([`evaluation_qwen35/`](../../evaluation_qwen35/)) for the **Qwen3.5-0.8B base + hybrid baselines** with the family-native `<tool_call>` / `<tool_response>` tag scheme. **M4.1 (2026-05-08)** swapped the placeholder flat-form `<tool_call>X</tool_call>` for the canonical Qwen3.5 nested-XML form (auto-injected by `apply_chat_template(..., tools=[QWEN35_SEARCH_TOOL])`); the rest of M4 is unchanged. This doc reflects the M4.1 state.
**Cluster (M4 eval)**: ALICE 1× A100-80GB (`gpu-short` + `gpu-a100-80g`).
**Source paths**: [`evaluation_qwen35/`](../../evaluation_qwen35/) (overlay copy of `evaluation_research/` with the M4 deltas), [`scripts/run_m4.sh`](../../scripts/run_m4.sh), [`scripts/sbatch_m4.sh`](../../scripts/sbatch_m4.sh), [`scripts/m4_download_models.sh`](../../scripts/m4_download_models.sh).

---

## 1. Headline Diff vs the M3 Pipeline (`evaluation_research/`)

| Dimension | M3 (`evaluation_research/`) | M4 (this doc, `evaluation_qwen35/`) |
|---|---|---|
| **Model family** | Qwen3-0.6B (hybrid) | **Qwen3.5-0.8B** (base + hybrid; both untrained) |
| **HF repos** | `Qwen/Qwen3-0.6B`, `pantomiman/Qwen3-0.6B-v0`, `pantomiman/Qwen3-0.6B-v0.1` | **`Qwen/Qwen3.5-0.8B`** + **`Qwen/Qwen3.5-0.8B-Base`** |
| **Action tag** | `<search>` / `</search>` (Search-R1 invention; not in pre-training) | **canonical Qwen3.5 nested-XML** `<tool_call><function=search><parameter=query>X</parameter></function></tool_call>` (CHAT_TEMPLATE.md §2; auto-injected by `apply_chat_template(..., tools=[QWEN35_SEARCH_TOOL])`) |
| **Observation tag** | `<result>` / `</result>` (M3 verl-legacy training rollout convention) | **`<tool_response>` / `</tool_response>`** (Qwen3.5-native vocab tokens 248066 / 248067), turn-bounded |
| **Prompt template** | `QWEN3_0_6B_TEMPLATE` (full protocol in system) | **`QWEN35_NATIVE_TEMPLATE`** (3-line role + 3-step protocol; tools schema auto-injected) + `QWEN35_SEARCH_TOOL` (OpenAI-style schema mirror of training-side `chat_template/tools.py:SEARCH_TOOL`) |
| **prompt_mode** | `qwen3` / `qwen3_p1_basic_w_ex` / `qwen3_p3_decide_no_ex` | **`qwen35`** / **`qwen35_native`** (alias) |
| **User message** | bare `{question}` | `Question: {question}` |
| **Final-answer wrap** | `<answer>The final answer is \[ \boxed{X} \]</answer>` (M3 reproducibility) | plain `<answer>X</answer>` |
| **Action stop tokens** | `[</search>, </answer>, <\|im_end\|>, <\|endoftext\|>]` | `[</tool_call>, </answer>, <\|im_end\|>, <\|endoftext\|>]` |
| **Result wrapper** | `" <result>\n{X}\n</result>"` (leading space; in-stream continuation) | `<\|im_end\|>\n<\|im_start\|>user\n<tool_response>\n{X}\n</tool_response><\|im_end\|>\n<\|im_start\|>assistant\n` (turn-bounded; mirrors training-side `format_docs_qwen_native`) |
| **`enable_thinking`** | True (Qwen3 hybrid) | **True for both hybrid and base** (M4.1 update 2026-05-09; earlier draft had False for base, flipped to give the base model open `<think>\n` reasoning space; the same render shape across variants also keeps the hybrid-vs-base comparison directly comparable) |
| **Per-mode budgets** | `max_search_turns=5`, `step_limit=8192`, `max_obs_length=256`, `retrieval_topk=5`, `generator_max_input_len=4096` | identical (M3 and M4 share the same budget shape for cross-family comparability) |
| **Retrieval text format** | raw `{contents}\n\n` joined and stripped | identical |
| **Retriever** | IVF-SQ8 × 8 workers (default); flat IP × 2 (paper-fidelity opt-in) | identical (project-root `corpus/`, `indexes/`, `models/`) |
| **SGLang launch flags** | `--context-length 8192 --dtype bfloat16 --trust-remote-code` | identical |
| **Eval venv** | `/home/s4374886/.conda/envs/evaluation_search_r1` (with `evaluation_research/` editable-installed) | **same conda env**; `evaluation_qwen35/flashrag/` resolves via cwd-precedence (script-dir is sys.path[0] when run via `cd evaluation_qwen35 && python run_eval.py`); no editable install needed |
| **Quick-eval (100 items)** | Not wired (see CODE_SETUP_m3 §10: `sample_num` was unused) | **`--test_sample_num 100 --random_sample True --seed 1`** plumbed through `run_eval.py` to FlashRAG's `Dataset.sample_num` (subsample at load time); `random.seed(config["seed"])` set in `run_eval.py` for determinism |

---

## 2. What's Unchanged (M3 14-fix audit holds verbatim)

The 14 alignment fixes catalogued in [`CODE_SETUP_m3.md`](CODE_SETUP_m3.md) §3 transfer to M4 unchanged:

- Project-root `corpus/`, `indexes/`, `models/` symlinks (#1)
- `pip install ninja` in the conda env (#2)
- `module load CUDA/12.4.0` + `source /etc/profile.d/lmod.sh` (#3)
- Single-brace template in `QWEN35_NATIVE_TEMPLATE` (#4 carries forward)
- `extract_solution` unwraps `\boxed{X}` from `<answer>...</answer>` content (#5; reused unchanged — still works on plain `<answer>X</answer>` since `\boxed{}` matching is optional in the regex)
- Turn-bounded `<\|im_end\|><\|im_start\|>user<tool_response>...</tool_response><\|im_end\|><\|im_start\|>assistant` envelope (#6 superseded by M4.1 — replaces the leading-space form with the canonical Qwen3.5 multi-turn shape)
- Raw `{contents}\n\n` retrieval text (#7; same)
- top-5 retrieval (#8; same)
- per-mode budgets (#9; same shape)
- `generator_max_input_len: 4096` (#10; same)
- `enable_thinking=True` for hybrid (#11) **and** base (M4.1 update 2026-05-09 — both variants get an open `<think>\n` generation prefix so the model reasons before each tool call)
- `SLURM_SUBMIT_DIR` for `REPO_ROOT` (#12; same)
- Retriever 1200 s readiness wait (#13; same as M3.1)
- HF arrow cache hard-link (#14; n/a if cache already warm from M3.1)

---

## 3. New M4-Specific Fixes (4 net-new deltas; §3.4 added on first Vast smoke)

### 3.1 `get_generator` short-circuits on `framework=sgl_remote` before VL detection

**Symptom:** Qwen3.5 ships with `vision_config` in its `config.json` (the family is the VL-co-pretrained one — `Qwen3_5ForConditionalGeneration`). FlashRAG's [`flashrag/utils/utils.py:50`](../../evaluation_qwen35/flashrag/utils/utils.py) checked `if all(["vision" not in key for key in model_config.keys()])` and routed to `HFMultiModalGenerator` whenever any `vision_*` config key was present. With M4 we want SGLang to do the inference (`framework=sgl_remote`); the in-process VL check should never run for the remote-HTTP path.

**Fix:** moved the `framework == "sgl_remote"` branch *before* the VL detection. The remote path returns `SGLRemoteGenerator(config, **params)` and skips the local config inspection altogether. Same fix benefits any future VL-flavoured model served via SGLang remote.

```python
# evaluation_qwen35/flashrag/utils/utils.py
if config['framework'] == 'openai':
    return getattr(...,"OpenaiGenerator")(config, **params)
# NEW: short-circuit before VL detection
if config["framework"] == "sgl_remote":
    return getattr(...,"SGLRemoteGenerator")(config, **params)
# (then the original VL / vLLM / hf / fschat dispatch)
```

Caught while inspecting the freshly-downloaded `Qwen/Qwen3.5-0.8B/config.json` on ALICE (commit `2be8e0a`).

### 3.2 Canonical Qwen3.5 nested-XML tool use + family-based dispatch in `active_pipeline.py`

**M4.1 design (replaces the M4-placeholder flat form).** Qwen3.5 was post-trained on the nested-XML tool-call form documented in [`docs/training/CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §2:

```
<tool_call>
<function=search>
<parameter=query>
QUERY TEXT
</parameter>
</function>
</tool_call>
```

The format spec is **auto-injected** by `apply_chat_template(..., tools=[QWEN35_SEARCH_TOOL])` — the chat template walks the tools schema and emits a `# Tools` block (function signature) + a verbatim format example + an `<IMPORTANT>` reminder, all before our system content. So our system prompt only needs the role intro + minimal protocol; the format spec is free.

The earlier M4 draft used a flat `<tool_call>X</tool_call>` form to keep the prose literal-identical to M3's `p1_basic_w_ex` and to avoid the ~270-token tools-schema preamble. That trade was incorrect: the flat form is **off-distribution** for Qwen3.5 (it asks the model to ignore its strongest tool-use prior), and the schema preamble is an in-distribution token cost the model was trained to handle. M4.1 swaps to the canonical form.

New parser ([`evaluation_qwen35/flashrag/search_r1/parser.py`](../../evaluation_qwen35/flashrag/search_r1/parser.py); mirror of training-side [`parsers.py:_RE_QWEN_QUERY`](../../training/src/environments/parsers.py)):

```python
def extract_tool_call_query(text):
    pattern = re.compile(
        r"<tool_call>.*?<parameter=query>\s*(.*?)\s*</parameter>.*?</tool_call>",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match: return "", "no_tool_call_tag"
    query = match.group(1).strip()
    return (query, "valid_tool_call_tag") if query else ("", "empty_tool_call_query")
```

Permissive: only the `<parameter=query>` body matters; the function name is not enforced (the registered schema only declares `search`, so any other name would fall to the corrective branch).

Family-based dispatch ([`evaluation_qwen35/flashrag/pipeline/active_pipeline.py`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py)):

```python
def _is_qwen3(mode):  return mode == 'qwen3'  or mode.startswith('qwen3_')   # NOT qwen35
def _is_qwen35(mode): return mode == 'qwen35' or mode.startswith('qwen35_')
def _is_qwen_family(mode): return _is_qwen3(mode) or _is_qwen35(mode)
```

The qwen35 branch additionally:
- Passes `tools=[QWEN35_SEARCH_TOOL]` to `apply_chat_template`.
- Builds the user message as `Question: {question}` (vs bare `{question}` for qwen3).
- Wraps the tool response in turn-bounded `<\|im_end\|><\|im_start\|>user<tool_response>...</tool_response><\|im_end\|><\|im_start\|>assistant\n` (vs leading-space continuation for qwen3).

Same per-mode budgets / retrieval text format / `enable_thinking` apply to qwen3 + qwen35 (the families share the 4096-budget / top-5 / 256-obs / 5-turn shape so cross-family comparison is apples-to-apples).

### 3.3 Quick-eval plumbing (`--test_sample_num` / `--random_sample` / `--seed`)

CODE_SETUP_m3 §10 noted that `sample_num` was *not respected* by the M3 search_r1 path. M4 wires the FlashRAG `test_sample_num` / `random_sample` / `seed` config keys through CLI flags so the smoke run can hit 100 random items per dataset deterministically:

```bash
--test_sample_num 100 --random_sample True --seed 1
```

Implementation ([`evaluation_qwen35/run_eval.py`](../../evaluation_qwen35/run_eval.py)):

1. Three CLI flags added; if `--test_sample_num` is set, `random_sample` and `seed` are also threaded into `config_dict` (and propagated to `Config`).
2. Before `get_dataset(config)`, call `random.seed(config["seed"])` so FlashRAG's `Dataset._load_data` uses a deterministic shuffle when `random_sample=True` (FlashRAG itself uses the global `random` module without seeding it).
3. `save_note` includes `_n100` so quick-eval results don't collide with full-sweep results in `results/<dataset>/`.

Full sweep recovers the original behaviour by simply omitting all three flags.

---

### 3.4 Qwen3.5 stop-token IDs in `active_pipeline.py` (Vast smoke discovery)

**Symptom:** First Vast smoke (M4.1, 2026-05-09) failed all 100 bamboogle items with `ValueError: stop_reason: stop, stop_matched: 248046`.

**Root cause:** SGLang reports the matched stop token as an integer id; the M3 pipeline only matched the Qwen3 ids `151643` (`<|endoftext|>`) and `151645` (`<|im_end|>`). Qwen3.5 has a new vocabulary: `<|endoftext|>` is `248044` and `<|im_end|>` is `248046`. Any natural turn-end (the model emits `<|im_end|>` after a final `<answer>...</answer>` block) hit the `else` branch and aborted the run.

**Fix:** added the Qwen3.5 ids to the stop-match branch in [`active_pipeline.py`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py); the existing string-token check on `<|im_end|>` / `<|endoftext|>` / `</answer>` is retained for environments where SGLang emits the matched-token string instead of the id:

```python
elif stop_reason == 'stop' and (
    # Qwen3 (M3) token ids
    stop_matched == 151643  # <|endoftext|>
    or stop_matched == 151645  # <|im_end|>
    # Qwen3.5 (M4) token ids; new vocab so different ids
    or stop_matched == 248044  # <|endoftext|>
    or stop_matched == 248046  # <|im_end|>
    or (isinstance(stop_matched, str) and stop_matched in {'<|im_end|>', '<|endoftext|>', '</answer>'})
):
```

Tokens verified via `AutoTokenizer.from_pretrained(eval/qwen3.5_0.8b).decode([id])`. The Qwen3.5 EOS is `<|im_end|>` (= 248046), distinct from `<|endoftext|>` (= 248044) which is the pad token.

---

## 4. Worktree-on-ALICE workflow (operational note, not a code fix)

ALICE's main checkout (`/zfsstore/user/s4374886/omega/reason_over_search/`) carries an active session for the M3.1 closure work on branch `research_v1`. To run the M4 smoke without disturbing that session, we created a **second git worktree** for `research_v2`:

```bash
# On ALICE:
MAIN=/zfsstore/user/s4374886/omega/reason_over_search
WT=/zfsstore/user/s4374886/omega/reason_over_search-m4
cd $MAIN
# LFS hooks are configured in the main repo but git-lfs binary is not on PATH on ALICE;
# disable the LFS filter for the one-shot worktree-add operation.
git -c core.hooksPath=/dev/null \
    -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false \
    worktree add $WT research_v2
cd $WT
# Symlink the heavy / gitignored artifacts that live in the main checkout:
for d in eval corpus indexes models data data_subsample logs; do
  [[ -e $MAIN/$d ]] && [[ ! -e $WT/$d ]] && ln -sfn $MAIN/$d $d
done
# Now sbatch from the worktree — SLURM_SUBMIT_DIR resolves to $WT;
# scripts/sbatch_m4.sh + run_m4.sh exist here; eval/ etc. resolve through symlinks.
sbatch --time=01:00:00 scripts/sbatch_m4.sh qwen3.5_0.8b 100
sbatch --time=01:00:00 scripts/sbatch_m4.sh qwen3.5_0.8b_base 100
```

The worktree shares `.git/objects/` with the main checkout (no extra disk for the git history). Branches are isolated (the main checkout stays on `research_v1`; the worktree stays on `research_v2`). When the M3.1 session is done and merges, we can `git worktree remove $WT` and run M4 from the main checkout.

---

## 4-bis. Vast.ai runtime (no SLURM, direct bash loop)

ALICE runs M4 via SLURM (`sbatch_m4.sh`); Vast doesn't have SLURM. We run the same eval loop directly under a `nohup`-wrapped bash. **No code in the pipeline changes**; only the venv paths, retriever-launch path, and SGLang-launch path are different.

**Vast venv layout** (from the `pantomiman/reason-over-search-v1:v1` image):

| Component | Path |
|---|---|
| Eval (FlashRAG + SGLang server + transformers) | `/venv/evaluation_search_r1` |
| Retriever (faiss-cpu + FastAPI) | `/venv/retriever` |
| Bootstrap helper (uv, etc.) | `/venv/main` |

(ALICE uses `/home/s4374886/.conda/envs/{evaluation_search_r1,retriever}` instead. Both `run_m4.sh` and `m4_download_models.sh` accept a `PY=...` env-var override; the auto-discovery path in `m4_download_models.sh` walks both Vast and ALICE candidates, so the same scripts run in both environments.)

### 4-bis.1 transformers upgrade for `qwen3_5` architecture support

**Symptom:** First smoke attempt failed at `flashrag/prompt/base_prompt.py:21` with `ValueError: The checkpoint you are trying to load has model type 'qwen3_5' but Transformers does not recognize this architecture`. The image-baked transformers was 4.57.1.

**Root cause:** `Qwen3_5ForConditionalGeneration` is a brand-new architecture (multi-modal-co-trained Qwen3.5; `model_type: qwen3_5` in `config.json`). Transformers 4.57.1 doesn't have it; the `CONFIG_MAPPING` lookup raises `KeyError('qwen3_5')`.

**Fix on Vast:**

```bash
/venv/evaluation_search_r1/bin/pip install 'transformers==5.7.0'
```

Verified the `/venv/retriever` env already shipped with transformers 5.7.0 (which loaded the config fine), so 5.7.0 is the floor that supports `qwen3_5`. The image's hard pin (`evaluation-search-r1` package requires `transformers<5,>=4.51.0`; SGLang 0.5.9 pins `transformers==4.57.1`) is reported as a dependency-resolver warning at install time; SGLang remains operational because the upgrade happens *after* SGLang has already loaded its weights into a separate process; runtime imports inside `flashrag` then succeed.

**Verified on Vast:**

```python
from transformers import AutoConfig
c = AutoConfig.from_pretrained("/workspace/reason_over_search/eval/qwen3.5_0.8b", trust_remote_code=True)
# OK: c.model_type == "qwen3_5"
```

**Forward note (image rebuild):** the image should bake `transformers==5.7.0` (or whatever floor supports `qwen3_5` at the time) and unpin SGLang's strict `==` requirement (or rebuild SGLang against the newer transformers). Tracked as a TODO in the docker README.

### 4-bis.2 Direct bash run loop (replacement for `sbatch_m4.sh`)

```bash
cd /workspace/reason_over_search
# Retriever already up on 3005 from the bootstrap; verify:
curl -sf http://127.0.0.1:3005/health

# Start SGLang for one variant (nohup'd):
export SGLANG_DISABLE_CUDNN_CHECK=1
nohup /venv/evaluation_search_r1/bin/python -m sglang.launch_server \
  --model-path /workspace/reason_over_search/eval/qwen3.5_0.8b \
  --host 127.0.0.1 --port 3000 \
  --tp 1 --context-length 8192 --dtype bfloat16 --trust-remote-code \
  > logs/m4_sglang_hybrid.log 2>&1 &

# Wait until SGLang is ready (~10 s for 0.8B on A100):
until curl -sf http://127.0.0.1:3000/health > /dev/null 2>&1; do sleep 5; done

# Run the smoke loop. PY must be passed via env for nohup'd subshells:
# just exporting in the parent doesn't propagate through nohup-bash-c
# the way --export=ALL does for sbatch.
nohup env PY=/venv/evaluation_search_r1/bin/python bash -c \
  'for DS in bamboogle nq triviaqa popqa hotpotqa 2wikimultihopqa musique; do
     bash scripts/run_m4.sh qwen3.5_0.8b $DS 1 /workspace/reason_over_search/data 100
   done' > logs/m4_smoke_hybrid.log 2>&1 &
```

When done, swap the `--model-path` to `eval/qwen3.5_0.8b_base` and rerun. **Kill the old SGLang first** (`pkill -f sglang.launch_server`); otherwise the new one fails to bind to port 3000 and the eval loop just talks to the previous variant.

The retriever is left running across both variants (single 8-worker IVF-SQ8 process; ~134 GB peak host RAM, fits comfortably in any A100/H100/H200 instance).

---

## 5. Variant dispatch + run recipe

| Variant | Path | `enable_thinking` | `prompt_mode` |
|---|---|---|---|
| `qwen3.5_0.8b` (hybrid) | `eval/qwen3.5_0.8b/` | True | `qwen35` |
| `qwen3.5_0.8b_base` | `eval/qwen3.5_0.8b_base/` | True | `qwen35` |

Quick eval (100 random items / dataset, seed 1):
```bash
sbatch --time=01:00:00 scripts/sbatch_m4.sh qwen3.5_0.8b 100
sbatch --time=01:00:00 scripts/sbatch_m4.sh qwen3.5_0.8b_base 100
```

Full sweep (51,713 items / variant):
```bash
sbatch scripts/sbatch_m4.sh qwen3.5_0.8b
sbatch scripts/sbatch_m4.sh qwen3.5_0.8b_base
```

Single-(variant, dataset, seed): [`scripts/run_m4.sh`](../../scripts/run_m4.sh) with same args + optional 5th positional `test_sample_num`.

---

## 6. Hardware sizing (unchanged from M3)

Same 8-worker IVF-SQ8 retriever (~134 GB RAM peak) + SGLang on 1× A100-80GB. CPU `--cpus-per-task=40`, `--mem=160g`. Smoke wall-clock target: ≤ 25 min total for both variants × 7 datasets × 100 items at 32 INFERENCE_MAX_WORKERS on a warm pipeline. Full sweep target: ~150 min / variant (M3 reference).

---

## 7. Pointers

- M3 alignment audit (the 14 fixes M4 inherits unchanged): [`CODE_SETUP_m3.md`](CODE_SETUP_m3.md) §3
- M4 milestone narrative: [`../milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md)
- M4 results table (when populated): `RESULTS_m4.md` (planned)
- Qwen3.5 chat template (verbatim): [`../training/CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §2
- Active recipe-ablation plan: [`../TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md)

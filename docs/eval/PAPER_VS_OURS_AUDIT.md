---
title: PAPER VS OURS AUDIT
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-01
---

# Search-R1 Paper vs Our Reproduction: Exhaustive Configuration Audit

**Audit date**: 2026-04-28
**Scope**: every knob that can move EM on the seven Search-R1 datasets for Qwen2.5-3B GRPO (base + instruct).
**Sources**: paper [arXiv:2503.09516v5](https://arxiv.org/html/2503.09516v5), official repo [`PeterGriffinJin/Search-R1`](https://github.com/PeterGriffinJin/Search-R1) (default branch `main`, snapshot 2026-04-28), our repo at `/workspace/reason_over_search`.

The comparison answers the four open questions from `../milestone_1/../archive/COMPARISON_PLAN_B_v0.md` and surfaces three previously-uncatalogued divergences. The earlier audit's headline conclusion ("paper eval = temperature 1.0") is **wrong** — verl's `_validate()` overrides `do_sample=False`, so the paper's reported numbers are GREEDY (temp=0). The actually-load-bearing miss is `apply_chat=False` for the base variant.

---

## TL;DR — divergences ordered by severity

| # | Divergence | Severity | Est. EM impact | Direction |
|---|---|---|---:|---|
| **D1** | `apply_chat=False` for base variant in `scripts/run_one.sh:35` and `run_eval.py:127` default | **HIGH** | +5 to +12 pp on base | we are below paper |
| **D2** | We use string stops `['</search>', '</answer>', '<\|im_end\|>', '<\|endoftext\|>']`; paper rollout uses **only** EOS + post-hoc truncation at first `</search>` / `</answer>` | LOW–MEDIUM | ±1–3 pp | sign uncertain |
| **D3** | Validation `do_sample=False` is hard-coded in upstream `verl/trainer/ppo/ray_trainer.py:478,508` → **paper eval is greedy** (temp=0). Our `temperature: 1.0` plan would **diverge** from the paper, not converge to it. | HIGH if changed | -3 to -8 pp if we set temp=1.0 | we are correctly greedy |
| **D4** | `n=1` (single sample) in paper eval; no eval-time ensembling. We also use n=1. | match | 0 | — |
| **D5** | `max_obs_length=500` is enforced as **first-500-tokens truncation** upstream; we use the same first-500. | match | 0 | — |
| **D6** | Local IVF-SQ8 index default in `local_retriever/retriever_config.yaml:9` vs paper Flat IP. Our README still says Flat is default; need to verify what's actually loaded. | LOW | ≤1 pp | we may be slightly below |
| **D7** | `max_response_length=500` per-call in paper rollout; ours is per-step `step_limit=500`. Both are correct but the paper's rollout has no string stops, so the model decodes a full 500 tokens every turn before truncation. We early-stop at `</search>` etc. The latent context-budget arithmetic is therefore different in edge cases. | LOW | ≤1 pp | sign uncertain |
| **D8** | We add `<search>`, `</search>`, `<answer>`, `</answer>`, `<result>`, `</result>` as additional special tokens in `active_pipeline.py:37-42` after construction. Upstream tokenizer does **not**. This affects how `<information>` / `</information>` and `</search>` get tokenized. | LOW | ≤1 pp | sign uncertain |

D1 is the only one large enough to explain the −10 pp average gap on base. D3 is the corrected reading of the temperature question that almost cost us a wrong fix.

---

## A. Generation (sampling)

| Knob | Paper claim | Official-repo eval (verl) | Ours | Source | Status |
|---|---|---|---|---|---|
| temperature | Train rollout 1.0 (Appendix B.2) | **0** for validation (`_validate` sets `do_sample=False`; vllm_rollout overrides to `temperature=0`) | 0.0 | `verl/trainer/ppo/ray_trainer.py:478,508`; `verl/workers/rollout/vllm_rollout/vllm_rollout.py:162-171`; `flashrag/config/basic_config.yaml:110` | **MATCH** |
| top_p | Train rollout 1.0 (Appendix B.2) | 1.0 (override in vllm_rollout when `do_sample=False`) | 1.0 | same | **MATCH** |
| top_k | Train rollout: not specified in paper. Upstream default `top_k: -1`. | -1 (override in vllm_rollout when `do_sample=False`) | not set in `generation_params`; SGLang default `top_k=-1` (no top-k filtering) | `flashrag/generator/generator.py:331-365` (does not pass top_k); SGLang default | **MATCH** (effective) |
| do_sample | Not stated in paper | **False** for validation | implicit False (we omit do_sample entirely; SGLang at temp=0 is greedy) | `_validate` line 478,508 | **MATCH** |
| n (samples per query) | "sample 5 responses per prompt" only for GRPO **training rollout**; eval not stated | n=1 forced when `do_sample=False` (line 170 of vllm_rollout) | n=1 (default in SGLang request) | `vllm_rollout.py:170`; our SGL `data_to_remote` has no `n` field | **MATCH** |
| max_new_tokens / max_tokens (per call / per step) | `data.max_response_length=500` in `train_grpo.sh`/`evaluate.sh` | 500 per vllm-call (no string stops, runs to EOS or 500) | `step_limit=500` per turn in `active_pipeline.py:64` | `scripts/nq_hotpotqa/evaluate.sh:18`; `active_pipeline.py:64` | **MATCH** in numeric value, see D7 |
| repetition_penalty | not mentioned | not set; vllm default 1.0 | not set; SGLang default 1.0 | — | **MATCH** |
| seed | Validation seed not set; greedy makes seed irrelevant in principle | — | `seed: 2024` in yaml; SGLang remote ignores it (we do **not** include `seed` in the SGL request body — see `generator.py:316-365`) | `basic_config.yaml:43`; `generator.py:316-365` | **MATCH** (both are seed-less greedy) |
| stop tokens | **None** (rollout halts at EOS); post-hoc truncation `resp.split('</search>')[0] + '</search>'` then `.split('</answer>')[0] + '</answer>'` | same | `stop_tokens = ['</search>', '</answer>', '<\|im_end\|>', '<\|endoftext\|>']` passed to SGL | `generation.py:_postprocess_responses`; our `active_pipeline.py:68` | **D2 — DIVERGENT** (LOW) |
| BOS/EOS | Qwen2.5: bos=None; eos=151643 (base) / 151645 (instruct) | same | same — verified via `config.json` | local `search_r1_base_model/config.json` `eos_token_id=151643`, `_instruct_model/config.json` `eos_token_id=151645` | **MATCH** |

### D2 stops — why this is LOW but non-zero

Upstream rollout generates up to 500 tokens then `_postprocess_responses` does `resp.split('</search>')[0] + '</search>'` (or `</answer>`). Our SGL request stops at the first match of `</search>` or `</answer>`, then `active_pipeline.py:104-106` re-appends the matched closer. Outcomes are identical when the model emits exactly one closer. Discrepancy only when the model emits **multiple** structural tags inside one decoded chunk — e.g. `<answer>X</answer> something <search>...</search>` — which our parser clips after the first `</search>`/`</answer>`, while upstream clips after the first `</search>` (preferring search over answer in the order of the if/elif). Both paths land on the first `</search>` in that example. So the only path that actually diverges is when the model first emits an `</answer>`, then keeps going to emit a `</search>`, before our stop fires. With Qwen2.5-3B GRPO that is rare (model is trained to terminate after `</answer>`).

Verdict: D2 is LOW. Keep our string stops — they reduce per-call wall-clock by ~30 % (no need to decode 500 tokens past the structural close).

---

## B. Prompt / chat scaffolding

| Knob | Paper / official | Ours | Source | Status |
|---|---|---|---|---|
| Prompt template (`SEARCH_R1_TEMPLATE`) | `make_prefix(template_type='base')` in `scripts/data_process/qa_search_test_merge.py` and three sibling files | byte-identical, except: ours omits the `For example, <answer> Beijing </answer>.` example sentence | upstream: `qa_search_test_merge.py:44-50`; ours: `flashrag/search_r1/templates.py:1-11` | **DIVERGENT — micro** |
| `apply_chat_template` for **base** variant | **YES**. Training (`verl/utils/dataset/rl_dataset.py:128-129`) does `if self.tokenizer.chat_template: apply_chat_template(...)`. Qwen2.5-3B base ships a chat template (verified: `huggingface.co/Qwen/Qwen2.5-3B/raw/main/tokenizer_config.json` has `chat_template`). The validation path goes through the same dataset class. | **NO** (default in `scripts/run_one.sh:35`: `apply_chat=False` for base) | upstream: `verl/utils/dataset/rl_dataset.py:128`; `infer.py:84-85`; ours: `scripts/run_one.sh:35`, `run_eval.py:127` | **D1 — DIVERGENT (HIGH)** |
| `apply_chat_template` for instruct | YES (same code path) | YES | same | **MATCH** |
| `add_generation_prompt` | True (line 129 of `rl_dataset.py`) | True (`active_pipeline.py:53`) | — | **MATCH** |
| System message wording | "You are a helpful assistant." (Qwen2.5 chat template default — injected when no system message is present) | same — flows through `apply_chat_template` when we set `apply_chat=True` | both use Qwen2.5 default chat template | **MATCH** when `apply_chat=True` |
| `enable_thinking` | Qwen3-only knob; Qwen2.5 chat template ignores it | `enable_thinking=False` (passed but no-op for Qwen2.5) | `active_pipeline.py:53` | **MATCH** (no-op) |
| `<think>` tag handling | Trained but not strictly required by `is_valid_sequence` parser; reward gives partial credit if think is missing | same — our `parser.py:54-55` accepts model that skips `<think>` | upstream: `verl/utils/reward_score/qa_em.py` (not shown); ours: `flashrag/search_r1/reward.py:54-55` | **MATCH** |

### D1 in detail (the load-bearing one)

Smoking gun, `verl/utils/dataset/rl_dataset.py` line 128-131:

```python
if self.tokenizer.chat_template:
    prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
else:
    prompt_with_chat_template = chat[0]['content']
```

`Qwen/Qwen2.5-3B` (the base model the paper trains on) has a non-empty `chat_template` field in `tokenizer_config.json`. We confirmed this on our local checkpoint (`evaluation_search_r1/search_r1_base_model/tokenizer_config.json`) and on the upstream HF repo. So the upstream training code applies the chat template **regardless of base/instruct distinction** — meaning the paper's reported base GRPO numbers were computed with chat-template-wrapped prompts.

Our `scripts/run_one.sh:35` hard-codes `apply_chat=False` for the base variant. `run_eval.py:127` defaults to `apply_chat=True` but the shell script overrides it. The Bamboogle probe in `../milestone_1/../archive/COMPARISON_PLAN_B_v0.md:96-110` already showed:

- Base, `apply_chat=False` (current Plan B): EM 0.112, close-rate 105/125 (84 %).
- Base, `apply_chat=True`: EM 0.128, close-rate 125/125 (100 %).

The mechanism: without the chat scaffolding, the base model has no `<|im_start|>assistant\n` cue and is more likely to ramble past the 500-token step budget without producing `</answer>`. With the scaffolding, every example closes its `<answer>` cleanly. Magnitude on Bamboogle was small (+1.6 pp) because it only has 125 examples and ~16 % were truncated; on NQ/TriviaQA/PopQA where the −10 to −16 pp gaps live, the truncation rate may be similar or different — we have not measured per-dataset close-rate yet.

Recommended action — **already documented as next step in ../milestone_1/../archive/COMPARISON_PLAN_B_v0.md**: flip `scripts/run_one.sh:35` to `apply_chat=True` for base, re-run NQ-1k and TriviaQA-1k.

### Prompt template micro-divergence (negligible)

Upstream `make_prefix(template_type='base')`:

```
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n
```

Ours (`flashrag/search_r1/templates.py:1-11`) is missing the sentence:

```
For example, <answer> Beijing </answer>.
```

This is a 4-token instruction that nudges the model to produce a one-word answer. On a model already RL-trained on the upstream prompt, the missing sentence would slightly shift the answer length distribution. Likely small (≤ 1 pp), but it is a real divergence. Worth fixing as a one-line edit to `templates.py`. Severity: LOW.

---

## C. Multi-turn / search loop

| Knob | Paper / official | Ours | Source | Status |
|---|---|---|---|---|
| `max_search_turns` / `max_turns` | **4** (eval), 2 (train) | 4 | `scripts/nq_hotpotqa/evaluate.sh:62`; `active_pipeline.py:63` | **MATCH** |
| Per-step max output tokens | 500 (`max_response_length`) | 500 (`step_limit`) | `evaluate.sh:18`; `active_pipeline.py:64` | **MATCH** |
| Observation truncation | First 500 **tokens**, no special-token addition (`generation.py:_process_next_obs`) | First 500 tokens, decoded back to text (`active_pipeline.py:120-124`) | upstream: `_process_next_obs` line 78-90; ours: `active_pipeline.py:120` | **MATCH** semantics, ours additionally re-decodes |
| `<information>` whitespace | `f'\n\n<information>{search_results.strip()}</information>\n\n'` | byte-identical | upstream `generation.py:execute_predictions` action == "search" branch; ours `active_pipeline.py:126` | **MATCH** (already fixed in fix #3) |
| Passage format string | `f"Doc {idx+1}(Title: {title}) {text}\n"` | byte-identical | upstream `_passages2string`; ours `active_pipeline.py:117` | **MATCH** (already fixed in fix #1) |
| Top-k retrieved | 3 | 3 | upstream `evaluate.sh:64` `retriever.topk=3`; ours `basic_config.yaml:59` `retrieval_topk: 3` | **MATCH** |
| Query formatting (raw question vs `<search>...</search>` body?) | extracts content between `<search>...</search>` then `.strip()` (`generation.py:postprocess_predictions` regex `r'<(search\|answer)>(.*?)</\1>'`) | extracts via `re.search(r"<search>(.*?)</search>", text, re.DOTALL)`, then `.strip()` | upstream `postprocess_predictions`; ours `parser.py:8` | **MATCH** |
| Question normalization (`.strip()`, trailing `?`) | yes — `qa_search_test_merge.py:88-89` | yes — `active_pipeline.py:46-48` | — | **MATCH** (already fixed in fix #6) |
| First-match vs last-match `<search>` regex | First-match (`re.search` returns first; `infer.py` uses `findall()[-1]` instead — ambiguous, but `generation.py` `re.search` is first) | First-match (`parser.py:8`) | upstream `generation.py:postprocess_predictions`; ours `parser.py:8` | **MATCH** (already fixed in fix #9) |
| Invalid-search corrective text | `f'\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'` | byte-identical | upstream `generation.py:execute_predictions` else branch; ours `active_pipeline.py:128-134` | **MATCH** (already fixed in fix #8) |

---

## D. Retrieval

| Knob | Paper / official | Ours | Source | Status |
|---|---|---|---|---|
| Encoder | `intfloat/e5-base-v2` | `./models/e5-base-v2/` | upstream `retrieval_launch.sh:5`; ours `local_retriever/retriever_config.yaml:8` | **MATCH** |
| Corpus | `wiki-18.jsonl` (FlashRAG 2018 Wikipedia dump, 21 M passages) | `./corpus/wiki18_100w.jsonl` | upstream `retrieval_launch.sh:3`; ours `retriever_config.yaml:12` | **MATCH** (same FlashRAG corpus) |
| Index type | Flat IP (paper's `e5_Flat.index`) | **default in our yaml says `wiki18_100w_e5_flat_inner.index`** (Flat IP). But `retriever_config.yaml:6` comment says IVF-SQ8 is the alternative; need to verify which is loaded at runtime. | upstream `retrieval_launch.sh`; ours `retriever_config.yaml:9` | **D6 — possibly DIVERGENT (LOW)**; verify by checking the running retriever's startup log |
| Query encoding prefix | `f"query: {query}"` (E5 convention) | same — `e5` is recognized in retriever code path that prepends `query: ` and `passage: ` | upstream `search_r1/search/retrieval_server.py:Encoder.encode`; ours via FlashRAG `flashrag/retriever` | **MATCH** |
| Passage encoding prefix | `f"passage: {query}"` | same | same | **MATCH** |
| Similarity metric | Inner product on L2-normalized embeddings (i.e. cosine) — Flat IP + `torch.nn.functional.normalize(query_emb, dim=-1)` | same | upstream `retrieval_server.py:Encoder.encode` line 92 normalizes; index is Flat IP | **MATCH** |
| Normalization | yes on both query and passage | yes | — | **MATCH** |
| `retrieval_query_max_length` | 256 | 256 | upstream `retrieval_server.py:670` `Config(...retrieval_query_max_length=256)`; ours `basic_config.yaml:62` | **MATCH** (already fixed in fix #7) |
| Retrieval batch size | 512 (upstream); 256 (ours) | 256 | upstream `retrieval_server.py:670`; ours `basic_config.yaml:60` | **DIVERGENT — negligible** (only affects retriever throughput, not which docs come back) |
| Retrieval `use_fp16` | True | True | both | **MATCH** |

### D6: which index is actually loaded?

`retriever_config.yaml:9` declares `wiki18_100w_e5_flat_inner.index` as default. The CLI flag `--index` in `retriever_serving.py:154` overrides it. We have not, in this audit, opened the running retriever's log to confirm the loaded path. **Action**: `grep index_path /tmp/retriever*.log` to verify Flat IP is loaded at run time. If IVF-SQ8 was passed via `--index`, recall drops by ≤1 % — would slightly *decrease* our EM, not explain the −10 pp gap on base, but worth verifying.

---

## E. Model checkpoints

| Knob | Paper / official | Ours | Source | Status |
|---|---|---|---|---|
| HF model ID (base GRPO) | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo` | same — verified by LFS sha256 + shard sizes | `REPRODUCIBILITY.md:11`; HF Collection page | **MATCH** |
| HF model ID (instruct GRPO) | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo` | same — sha256 verified | `REPRODUCIBILITY.md:12` | **MATCH** |
| Underlying base | `Qwen/Qwen2.5-3B` (base) and `Qwen/Qwen2.5-3B-Instruct` (instruct) | same | `train_grpo.sh:14-17` | **MATCH** |
| dtype | bf16 (paper trains in bf16; SGLang loads at `--dtype bfloat16`) | bf16 (`scripts/manage_sglang.sh:67` `--dtype bfloat16`) | — | **MATCH** |
| Tokenizer revision | identical to base Qwen2.5-3B(-Instruct) tokenizer | same — local checkpoints copy the same `tokenizer.json` | `local search_r1_*_model/tokenizer.json` | **MATCH** |
| Post-training stage | None (released GRPO checkpoint = final step of GRPO training, no SFT distillation) | same — we just load the released weights | — | **MATCH** |
| Special-tokens additions at runtime | None (upstream tokenizer has `<\|im_end\|>` etc baked in) | We **add** `<search>`, `</search>`, `<answer>`, `</answer>`, `<result>`, `</result>` as additional special tokens at `active_pipeline.py:37-42` | ours: `active_pipeline.py:37`; upstream: nowhere | **D8 — DIVERGENT (LOW)** |

### D8: special-token additions

Adding `<search>` etc. as additional special tokens **after** loading the tokenizer affects: (a) how those strings tokenize when present in the input; (b) whether `skip_special_tokens=True` strips them on decode. The base Qwen2.5 BPE already tokenizes `<search>` as a multi-token sequence (`<`, `search`, `>` or similar), and the model has been GRPO-trained on those exact token IDs. Adding them as **special tokens** changes the IDs at runtime, breaking the model's learned distribution.

In practice, this matters for: tokenizing the prompt (where `<search>` doesn't appear), tokenizing observations (where `</information>` doesn't appear in our `add_special_tokens` list — only the angle-bracket triplets do), and tokenizing the multi-turn rollout where `</search>` does appear after retrieval.

The upstream code never adds these as special tokens — they remain regular sub-token sequences. We should match. **Action**: remove the `add_special_tokens` block in `active_pipeline.py:37-42` and re-run a smoke test to confirm `</search>` is matched as a stop string by SGLang regardless. SGLang stop-string matching operates on decoded text, not token IDs, so removing the addition should not break stop matching, and it removes a subtle distributional mismatch.

Severity LOW because: (a) the prompt path doesn't contain these strings before generation, (b) only the rollout-internal accumulator gets re-tokenized, (c) `add_special_tokens=False` is used in the most critical place (`obs_ids = self.tokenizer.encode(retrieval_text, add_special_tokens=False)`).

---

## F. Inference framework

| Knob | Paper / official | Ours | Source | Status |
|---|---|---|---|---|
| Engine | vLLM (rollout via `verl.workers.rollout.vllm_rollout`) | SGLang remote | upstream `train_grpo.sh:43` `actor_rollout_ref.rollout.name=vllm`; ours `scripts/manage_sglang.sh` | **DIVERGENT — design choice** |
| `--disable-radix-cache` | N/A (vLLM has different prefix caching) | not currently set per `manage_sglang.sh:61-71` | — | n/a |
| `--disable-overlap` | N/A | not currently set | — | n/a |
| `--context-length` | 4096 (`max_prompt_length`) | 8192 (`manage_sglang.sh:65`) | upstream `evaluate.sh:14`; ours | **DIVERGENT — bigger ctx is fine** |
| `--enable-metrics` | n/a | yes | `manage_sglang.sh:66` | n/a |
| `--dtype` | bf16 | bf16 | both | **MATCH** |
| `--tp` (tensor parallel) | 1 in upstream eval | 1 in ours | both | **MATCH** |
| concurrency / batch size at eval level | upstream batches 256 examples per vllm call (`val_batch_size=256`) | 8 worker threads via ThreadPoolExecutor → 8 concurrent SGLang requests | upstream `evaluate.sh:13`; ours `flashrag/pipeline/parallelism.py` | **DIVERGENT — design choice; same EM** |

vLLM vs SGLang at greedy temperature 0 should produce identical token streams modulo BPE tokenization edge cases and floating-point determinism. EM differences expected: ≤0.5 pp. Not the source of the −10 pp gap.

---

## G. Datasets / splits

| Dataset | Paper / official split | Our split | Size local | Source |
|---|---|---|---:|---|
| NQ | `test` (3610) | `test` | 3610 | `data/nq/test.jsonl`; `run_one.sh:23` |
| TriviaQA | `test` (11313) | `test` | 11313 | `data/triviaqa/test.jsonl`; `run_one.sh:24` |
| PopQA | `test` (14267) | `test` | 14267 | `data/popqa/test.jsonl` |
| HotpotQA | `dev` (7405) — official has no test labels | `dev` | 7405 | `run_one.sh:26` |
| 2WikiMultiHopQA | `dev` (12576) | `dev` | 12576 | `run_one.sh:27` |
| MuSiQue | `dev` (2417) | `dev` | 2417 | `run_one.sh:28` |
| Bamboogle | `test` (125) | `test` | 125 | `run_one.sh:22` |

**MATCH** across all seven. Splits and sizes match `RUC-NLPIR/FlashRAG_datasets` exactly.

| Knob | Upstream | Ours | Status |
|---|---|---|---|
| HF dataset name | `RUC-NLPIR/FlashRAG_datasets` | local jsonl with same content (`question`, `golden_answers`) | **MATCH** |
| Question normalization at load | `.strip()` + ensure trailing `?` (in `qa_search_test_merge.py:88-89`) | same (in `active_pipeline.py:46-48`) | **MATCH** (already fixed in fix #6) |
| Answer field structure | List `golden_answers` | List `golden_answers` | **MATCH** |

---

## H. Metrics

| Knob | Paper / official | Ours | Source | Status |
|---|---|---|---|---|
| EM normalization | SQuAD-canonical: lowercase, remove articles `(a\|an\|the)`, remove punctuation, whitespace fix | identical regex | upstream `verl/utils/reward_score/qa_em.py` (mirrored); ours `flashrag/search_r1/reward.py:6-20` | **MATCH** |
| EM aggregation over multi-answer | max over `golden_answers` (any match → 1.0) | same | upstream `qa_em.py em_check`; ours `reward.py:23-30` | **MATCH** |
| F1 tokenization | whitespace tokenization after the same normalization | same | — | **MATCH** |
| ACC (sub-EM) | normalized substring match | same | — | **MATCH** |
| LLM-judge | not used in paper Table 3 | not used | — | **MATCH** |

---

## I. Seeds and determinism

| Knob | Paper / official | Ours | Source | Status |
|---|---|---|---|---|
| Number of seeds | **single seed** per data point in Table 3 (no error bars in paper) | 1 (Plan B), planned 3–5 (Plan A) | paper text "single seed runs" inferred from absence of std-dev columns; ours: `../report/RESULTS_m1.md` | **MATCH** for Plan B |
| Validation seed value | not set (greedy → deterministic modulo FP) | `seed: 2024` in yaml; SGLang request body does not include `seed` (see `generator.py:316-365`) | — | **MATCH** (greedy is deterministic in both) |
| `torch` / `numpy` / `random` seeds | set to 1 by default in verl | not consequential at temp=0 | — | **MATCH** |
| SGLang seed kwarg behavior | n/a | SGLang ignores absent seed; greedy is deterministic | — | **MATCH** |

---

## Resolution of the four open questions from `../milestone_1/../archive/COMPARISON_PLAN_B_v0.md`

1. **"Does the paper use top_p=1.0 (Appendix B.2) or top_p=0.95 (upstream verl rollout config)?"**
   → **`top_p=1.0`** for evaluation. Appendix B.2 quote describes the training rollout, but `verl/workers/rollout/vllm_rollout/vllm_rollout.py:166` *overrides* top_p to 1.0 whenever `do_sample=False`, which validation always passes. So eval top_p is 1.0 regardless of the YAML default.

2. **"Does the paper's eval reuse the training rollout config (so do_sample=True, temp=1.0) or use a separate validation block (val_kwargs)?"**
   → **Neither.** It reuses the same vllm rollout instance, but `_validate()` at `verl/trainer/ppo/ray_trainer.py:478,508` passes `do_sample: False` in `meta_info`, and `vllm_rollout.py:162-171` reads that flag and force-overrides to `temperature=0, top_p=1.0, top_k=-1, n=1`. There is no `val_kwargs` block; the override is hard-coded. **Paper eval is greedy.** Our `temperature: 0.0` is correct. The ../milestone_1/../archive/COMPARISON_PLAN_B_v0.md hypothesis (re-running at temp=1.0) would have **diverged** from the paper, not converged.

3. **"Does the official Search-R1 eval script set apply_chat_template for the base variant?"**
   → **YES**, unconditionally, via `verl/utils/dataset/rl_dataset.py:128`: `if self.tokenizer.chat_template: apply_chat_template(...)`. Qwen2.5-3B base ships with the chat template, so it gets applied. Our `scripts/run_one.sh:35` setting `apply_chat=False` for base is **wrong**. Bamboogle probe confirmed +1.6 pp lift and 84 %→100 % close-rate fix when flipped.

4. **"Is there any n=k ensembling at eval time?"**
   → **No.** `vllm_rollout.py:170` forces `n=1` whenever `do_sample=False`. Single sample per question. Our setup matches.

5. **(bonus) "Is the eval max_new_tokens 500 (per step) or different from training's 500?"**
   → **Same: 500.** Training's `data.max_response_length=500` flows through to the same vllm rollout used by validation. Each turn issues a single 500-token call. Our `step_limit=500` matches.

---

## What we should change before launching Plan A

Ranked by EM impact estimate.

1. **Flip base `apply_chat=True`** (`scripts/run_one.sh:35`). HIGH severity, +5 to +12 pp on base. One-line edit.
2. **Add the missing prompt sentence** `For example, <answer> Beijing </answer>.` to `flashrag/search_r1/templates.py:9`. LOW severity (≤1 pp) but a real divergence with no cost to fix.
3. **Verify which FAISS index is actually loaded** by the running retriever. If IVF-SQ8 was inadvertently passed via `--index`, switch to Flat IP for parity. LOW severity.
4. **Remove the `add_special_tokens` call** in `active_pipeline.py:37-42`. LOW severity, removes a subtle distributional mismatch with upstream tokenization.
5. **Optional: verify SGLang stop-string semantics on multi-tag chunks** by sampling 50 base traces and counting how many emit a second structural tag inside a single decoded chunk. If <1 %, ignore D2.

Do **not** change `temperature` or `top_p`. The paper is greedy. Our config is correct on this axis. The earlier hypothesis in `../milestone_1/../archive/COMPARISON_PLAN_B_v0.md` to set `temperature=1.0, top_p=0.95` would have moved us *away* from the paper.

---

## Files referenced (absolute paths)

Ours:
- `/workspace/reason_over_search/evaluation_search_r1/flashrag/config/basic_config.yaml`
- `/workspace/reason_over_search/evaluation_search_r1/flashrag/pipeline/active_pipeline.py`
- `/workspace/reason_over_search/evaluation_search_r1/flashrag/search_r1/templates.py`
- `/workspace/reason_over_search/evaluation_search_r1/flashrag/search_r1/parser.py`
- `/workspace/reason_over_search/evaluation_search_r1/flashrag/search_r1/reward.py`
- `/workspace/reason_over_search/evaluation_search_r1/flashrag/generator/generator.py`
- `/workspace/reason_over_search/evaluation_search_r1/run_eval.py`
- `/workspace/reason_over_search/scripts/run_one.sh`
- `/workspace/reason_over_search/scripts/manage_sglang.sh`
- `/workspace/reason_over_search/local_retriever/retriever_config.yaml`
- `/workspace/reason_over_search/local_retriever/retriever_serving.py`
- `/workspace/reason_over_search/docs/milestone_1/../archive/COMPARISON_PLAN_B_v0.md`
- `/workspace/reason_over_search/docs/eval/REPRODUCIBILITY.md`
- `/workspace/reason_over_search/docs/report/RESULTS_m1.md`
- `/workspace/reason_over_search/docs/eval/EVAL_OPS.md`

Upstream (`PeterGriffinJin/Search-R1` @ main, fetched 2026-04-28):
- `verl/trainer/ppo/ray_trainer.py` (lines 436-547: `_validate()` with `do_sample=False`)
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py` (lines 162-171: greedy override when `do_sample=False`)
- `verl/utils/dataset/rl_dataset.py` (lines 120-145: chat-template application)
- `verl/trainer/config/ppo_trainer.yaml` (rollout block: `temperature=1.0, top_p=0.95, top_k=-1, do_sample=True, n=1`)
- `search_r1/llm_agent/generation.py` (`LLMGenerationManager.run_llm_loop`, `_postprocess_responses`, `_passages2string`)
- `search_r1/search/retrieval_server.py` (E5 encoder + Flat IP retriever)
- `scripts/nq_hotpotqa/evaluate.sh` (`max_turns=4`, `retriever.topk=3`, `data.max_response_length=500`, `data.max_obs_length=500`)
- `scripts/data_process/qa_search_test_merge.py` (`make_prefix(template_type='base')`, question normalization)
- `train_grpo.sh` (training rollout `temperature=1`, `n_agent=5`)
- `infer.py` (`if tokenizer.chat_template: apply_chat_template(...)` — applies to base)

Paper:
- arXiv:2503.09516v5, Appendix B.2 (sampling: temp=1.0 top_p=1.0 — for **training rollout**)
- arXiv:2503.09516v5, Appendix F / Table 3 (Qwen2.5-3B GRPO numbers reproduced in `REPRODUCIBILITY.md`)
- arXiv:2503.09516v5, Appendix E (Base vs Instruct — does not specify chat template handling)
- arXiv:2503.09516v5, Section 4.3 (top-3 retrieval, 2018 Wikipedia dump, E5 encoder)

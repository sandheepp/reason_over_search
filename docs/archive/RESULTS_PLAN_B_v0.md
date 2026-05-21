---
title: RESULTS PLAN B v0
tags: []
source: internal
created: 2026-04-28
updated: 2026-04-28
---

# Search-R1 evaluation results

_Source: `/workspace/reason_over_search/evaluation_search_r1/results` (14 runs)_

## Per-seed scores

### EM

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.112 | 0.112 | — | 1 |
| bamboogle | instruct | 0.360 | 0.360 | — | 1 |
| nq | base | 0.316 | 0.316 | — | 1 |
| nq | instruct | 0.399 | 0.399 | — | 1 |
| triviaqa | base | 0.421 | 0.421 | — | 1 |
| triviaqa | instruct | 0.539 | 0.539 | — | 1 |
| popqa | base | 0.309 | 0.309 | — | 1 |
| popqa | instruct | 0.412 | 0.412 | — | 1 |
| musique | base | 0.034 | 0.034 | — | 1 |
| musique | instruct | 0.149 | 0.149 | — | 1 |
| 2wikimultihopqa | base | 0.207 | 0.207 | — | 1 |
| 2wikimultihopqa | instruct | 0.353 | 0.353 | — | 1 |
| hotpotqa | base | 0.201 | 0.201 | — | 1 |
| hotpotqa | instruct | 0.354 | 0.354 | — | 1 |

### F1

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.172 | 0.172 | — | 1 |
| bamboogle | instruct | 0.451 | 0.451 | — | 1 |
| nq | base | 0.395 | 0.395 | — | 1 |
| nq | instruct | 0.481 | 0.481 | — | 1 |
| triviaqa | base | 0.475 | 0.475 | — | 1 |
| triviaqa | instruct | 0.628 | 0.628 | — | 1 |
| popqa | base | 0.343 | 0.343 | — | 1 |
| popqa | instruct | 0.460 | 0.460 | — | 1 |
| musique | base | 0.079 | 0.079 | — | 1 |
| musique | instruct | 0.221 | 0.221 | — | 1 |
| 2wikimultihopqa | base | 0.253 | 0.253 | — | 1 |
| 2wikimultihopqa | instruct | 0.423 | 0.423 | — | 1 |
| hotpotqa | base | 0.269 | 0.269 | — | 1 |
| hotpotqa | instruct | 0.459 | 0.459 | — | 1 |

### ACC

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.120 | 0.120 | — | 1 |
| bamboogle | instruct | 0.376 | 0.376 | — | 1 |
| nq | base | 0.350 | 0.350 | — | 1 |
| nq | instruct | 0.439 | 0.439 | — | 1 |
| triviaqa | base | 0.467 | 0.467 | — | 1 |
| triviaqa | instruct | 0.616 | 0.616 | — | 1 |
| popqa | base | 0.325 | 0.325 | — | 1 |
| popqa | instruct | 0.452 | 0.452 | — | 1 |
| musique | base | 0.040 | 0.040 | — | 1 |
| musique | instruct | 0.173 | 0.173 | — | 1 |
| 2wikimultihopqa | base | 0.224 | 0.224 | — | 1 |
| 2wikimultihopqa | instruct | 0.395 | 0.395 | — | 1 |
| hotpotqa | base | 0.211 | 0.211 | — | 1 |
| hotpotqa | instruct | 0.389 | 0.389 | — | 1 |

## Grand averages

### Grand average EM across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.229 | 7 |
| instruct | 0.367 | 7 |

### Grand average F1 across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.284 | 7 |
| instruct | 0.446 | 7 |

### Grand average ACC across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.248 | 7 |
| instruct | 0.406 | 7 |

---
title: SMOKE RESULTS 2026 05 06
tags: []
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# Smoke Test Results — Phase 2 First-Pass Mechanics

> **Canonical smoke-results doc.** Latest sample dump = the v4 (seed=43) run; the timing / wall-clock / cost analysis at the bottom comes from the **2026-05-02 small-shape run** (40-traj/combo). Both archived; see "Provenance + archive" below.

Run on Vast.ai 1× A100 80 GB, image `pantomiman/reason-over-search-v1:v1`. Retriever: IVF-SQ8 + 8 workers (the flat IP default times out under training rollout load). Sample dump shape (this file's body): 8 outer steps × 8 prompts × group=5 = 320 trajectories per combo (40 per step jsonl line). **Seed = 43** (v2 used seed=42); same code, same data, same hyperparameters — only the seed differs.

## Provenance + archive

| File | Date | Shape | What it has | Where it lives |
|---|---|---|---|---|
| **SMOKE_RESULTS_2026-05-06.md** *(this file)* | 2026-05-06 | 8×8×5 = 320 traj/combo, seed=43 | latest sample dump + v2-vs-v4 seed-variance comparison + appended timing/cost analysis | [`docs/training/`](.) |
| SMOKE_RESULTS_2026-05-04_v2_seed42.md | ~2026-05-04 | 8×8×5 = 320 traj/combo, seed=42 | sample dump only (no timing) | [`docs/archive/training/`](../archive/training/SMOKE_RESULTS_2026-05-04_v2_seed42.md) |
| SMOKE_RESULTS_2026-05-02_smoke_shape.md | ~2026-05-02 | 2×4×5 = 40 traj/combo | **source of all timing / wall-clock / cost numbers cited throughout the project** + GPU utilisation trace + bottleneck inventory | [`docs/archive/training/`](../archive/training/SMOKE_RESULTS_2026-05-02_smoke_shape.md) |

The Timing / GPU / Bottlenecks / Vast-sizing sections at the bottom of this file are **lifted verbatim** from the 2026-05-02 run (40-traj smoke shape, single 1× A100 80GB SXM box). Cited line ranges in [`docs/research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md), [`docs/research/INTEGRATION_GUIDE.md`](../research/INTEGRATION_GUIDE.md), and [`docs/research/PARADIGM_REVIEW.md`](../research/PARADIGM_REVIEW.md) point at this file's "Timing, GPU utilization, and bottlenecks" section below.

**Required smoke overrides** (Qwen3.5 + multimodal + Mamba layers):

```
policy.sequence_packing.enabled=false  policy.dynamic_batching.enabled=true  policy.train_micro_batch_size=2
```

## Summary

| Combo | Run dir | Mean reward | EM hits | Has-answer % | Avg retrieval calls | Notes |
|---|---|---:|---:|---:|---:|---|
| base × qwen_native | `/workspace/reason_over_search/logs/smoke_v4_base_qwen_native` | 0.107 | 0/320 | 47% | 1.9 | — |
| base × paper | `/workspace/reason_over_search/logs/smoke_v4_base_paper` | 0.085 | 2/320 | 55% | 0.2 | ✓ |
| hybrid × qwen_native | `/workspace/reason_over_search/logs/smoke_v4_hybrid_qwen_native` | 0.248 | 0/320 | 65% | 2.0 | — |
| hybrid × paper | `/workspace/reason_over_search/logs/smoke_v4_hybrid_paper` | 0.059 | 0/320 | 56% | 0.8 | — |

## Per-step mean reward (trend)

| Combo | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base × qwen_native | 0.125 | 0.100 | 0.088 | 0.130 | 0.095 | 0.028 | 0.183 | 0.105 |
| base × paper | 0.073 | 0.103 | 0.078 | 0.068 | 0.085 | 0.095 | 0.115 | 0.063 |
| hybrid × qwen_native | 0.258 | 0.298 | 0.198 | 0.283 | 0.310 | 0.093 | 0.295 | 0.253 |
| hybrid × paper | 0.063 | 0.083 | 0.045 | 0.045 | 0.073 | 0.063 | 0.045 | 0.060 |

## base × qwen_native

- trajectories: 320 (40/step × 8 steps)
- reward distribution: EM-hit (≥0.8): 0 | format-only (0.1–0.3): 120 | zero: 173
- has-`<answer>`: 151/320, paper-format-valid: 120/320
- avg retrieval calls per traj: 1.93 (<tool_call>)

### Sample 1 — step 1, idx 18 — reward=0.800, retrieval-calls=1, input_len=1242 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…869 chars elided…]

easoning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: who is the longest winning streak in nba?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about the longest winning streak in the NBA. I need to use the search tool to find external knowledge about this specific information. I should formulate a relevant search query to get factual knowledge about NBA winning streaks.
</think>

I'll search for information about the longest winning streak in the NBA.

<tool_call>
<function=search>
<parameter=query>
longest winning streak NBA team
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "Los Angeles Lakers"
decade. The Lakers hold the record for NBA's longest winning streak, 33 straight games, set during the 1971–72 season. 21 Hall of Famers have played for Los Angeles, while four have coached the team. Four Lakers—Abdul-Jabbar, Johnson, O'Neal, and Bryant—have won the NBA

[…2134 chars elided…]

 the Warriors' attempt at a home streak.

However, the Warriors' record is for consecutive regular-season games, not necessarily consecutive games where they actually won (a streak). And the Lakers' 33-game win streak would be more representative of actual winning streaks, not just games played.

Looking at the search results, I also see that some results cut off at the beginning, but I have enough information to answer the question. The best answer would be the Lakers' 33-game winning streak, as they're the longest winning streak in NBA history.
</think>

<answer> Los Angeles Lakers </answer>
```

### Sample 2 — step 1, idx 25 — reward=0.800, retrieval-calls=1, input_len=1094 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…879 chars elided…]

nside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the indiana jones ride open at disneyland?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about when "Indiana Jones and the Temple of Doom" (I believe what they mean by "Indiana Jones ride" is the specific attraction at DisneyLand) opened at Disneyland. I need to use the search tool to find accurate information about this opening date.

Let me search for this information.
</think>

<tool_call>
<function=search>
<parameter=query>
Indiana Jones and the Temple of Doom attraction opening date Disneyland
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "Indiana Jones Adventure"
Indiana Jones Adventure The Indiana Jones Adventure is an enhanced motion vehicle dark ride attraction at Disneyland and Tokyo DisneySea, based on the ""Indiana Jones"" film series. Guests accompany intrepid archaeologist Dr. Indiana Jones on a

[…1587 chars elided…]

rity guests were George Lucas, Michael Eisner
</tool_response><|im_end|>
<|im_start|>assistant
<think>
Based on the search results, I can see that "Indiana Jones Adventure The Indiana Jones Adventure" debuted on Disneyland with the premiere on March 4, 1995. I can see from Doc 1 specifically that the attraction premiered at Disneyland on March 3, 1995, and officially opened to the general public on March 4, 1995. This matches with what the user is asking for - when the attraction opened at Disneyland.

Let me provide the answer concisely as requested.
</think>

<answer> March 4, 1995 </answer>
```

### Sample 3 — step 1, idx 28 — reward=0.800, retrieval-calls=1, input_len=1082 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…879 chars elided…]

nside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the indiana jones ride open at disneyland?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about when the Indiana Jones ride opened at Disneyland. This is a factual question about a specific public establishment. The search tool is designed for Wikipedia-style searches and looking up general information. For this query, I should use the search tool to find relevant factual information about the Indiana Jones ride at Disneyland's opening date.
</think>

I'll search for information about the Indiana Jones ride opening at Disneyland.

<tool_call>
<function=search>
<parameter=query>
Indiana Jones ride opening date Disneyland theme park
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "Indiana Jones Adventure"
Indiana Jones Adventure The Indiana Jones Adventure is an enhanced motion vehicle dark ride attraction at Disn

[…1672 chars elided…]

nhancements to lighting, paint, figure animation and other effects. In 2013, the idol of Mara that appears in the Hall of Promise scene was updated with projection mapping 
…[truncated]
</tool_response><|im_end|>
<|im_start|>assistant
<think>
I can see the results from the search. From Doc 1, I can see that the Indiana Jones Adventure opened "Temple of the Forbidden Eye at Disneyland in Anaheim, California on March 3, 1995, and opened to the general public on March 4, 1995." This is exactly what the user is asking about - the opening date at Disneyland.
</think>

<answer>March 4, 1995</answer>
```

### Sample 4 — step 1, idx 29 — reward=0.800, retrieval-calls=1, input_len=1044 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…879 chars elided…]

nside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the indiana jones ride open at disneyland?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking for information about when the Indiana Jones: The Ride opened at Disneyland (which is now called Disneyland). This is a factual question about a movie/ride opening date. I can use the search tool to find this information using Wikipedia as a resource.
</think>

<tool_call>
<function=search>
<parameter=query>
Indiana Jones The Ride opening date Disneyland
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "Indiana Jones Adventure"
Indiana Jones Adventure The Indiana Jones Adventure is an enhanced motion vehicle dark ride attraction at Disneyland and Tokyo DisneySea, based on the ""Indiana Jones"" film series. Guests accompany intrepid archaeologist Dr. Indiana Jones on a turbulent quest, aboard military troop transport vehicles

[…1457 chars elided…]

n of the attraction. The Temple of the Forbidden Eye premiered on March 3, 1995. Among the invited celebrity guests were George Lucas, Michael Eisner
</tool_response><|im_end|>
<|im_start|>assistant
<think>
Based on the search results, I found that "Indiana Jones Adventure" opened at Disneyland on March 4, 1995. The search results show multiple documents confirming this date. I can provide the direct answer without detailed illustrations since it's just a date.
</think>

Indiana Jones Adventure - The Indiana Jones Adventure opened at Disneyland on March 4, 1995.

<answer>March 4, 1995</answer>
```

### Sample 5 — step 1, idx 0 — reward=0.000, retrieval-calls=4, input_len=2278 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…864 chars elided…]

uct reasoning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the beatles come to the us?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking when The Beatles came to the US. This is a factual question about music history. I should use the search tool to find the relevant information.
</think>

I'll search for information about The Beatles' arrival in the US.

<tool_call>
<function=search>
<parameter=query>
The Beatles arrival US college summer stock movies
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "College circuit"
the ouvre of Ken Russell. Often the films are of no discernible artistic merit at all, but instead appeal to the culture of college life, such as in the case of the Beatles' Magical Mystery Tour. College circuit College circuit is a form of motion picture distribution where old films as well as new ones are shown on college campuses, usually in 

[…6575 chars elided…]

at houses the Beatles' Apple Corps. Harrison's earliest Beatles compositions were published by Northern Songs, the company set up in 1963 by Dick James and Brian Epstein on behalf of John Lennon and Paul McCartney, who were the main
</tool_response><|im_end|>
<|im_start|>assistant
<think>
I'm still getting information about The Beatles' music career rather than when they came to the US. Let me try a different approach and search for their first gigs or base in North America.
</think>

<tool_call>
<function=search>
<parameter=query>
The Beatles first US gig
</parameter>
</function>
</tool_call>
```

### Sample 6 — step 1, idx 4 — reward=0.000, retrieval-calls=2, input_len=2181 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…864 chars elided…]

uct reasoning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the beatles come to the us?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about when The Beatles came to the US. This is a factual question that can be answered by using the search tool to find external knowledge about this historical event.
</think>

<tool_call>
<function=search>
<parameter=query>
Beatles US opening night when did they come to the United States
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "The Beatles"
the Beatles left the United Kingdom with an estimated 4,000 fans gathered at Heathrow, waving and screaming as the aircraft took off. Upon landing at New York's John F. Kennedy Airport, an uproarious crowd estimated at 3,000 greeted them. They gave their first live US television performance two days later on ""The Ed Sullivan Show"", watched by approximately 73 million viewers 

[…6181 chars elided…]

reak in Glasgow, they departed again for the US break, this time in Chicago, and ended up in Hollywood by 14 January 1965 after a delay. For their first home show at Silver Studio in Los Angeles on 10 January 1965, JP had the studio lines perfectly worded so that the band could choose the opening song themselves, leading to "Blackbird" becoming their signature opening number, despite John Lennon's initial preference for "Please Please Me".

The most famous early US performance was indeed the Shea Stadium show on **15 August 1965**, which set records for both attendance and revenue generation.

```

### Sample 7 — step 1, idx 7 — reward=0.000, retrieval-calls=1, input_len=1227 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…872 chars elided…]

oning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: how much did the gnp drop from 1929 to 1932?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about a specific economic fact - the drop in real GDP from 1929 to 1932. This is a historical economic question that would require factual knowledge about the Great Depression in the United States. 

I should use the search tool to find this specific information. Let me search for "Real GDP drop United States 1929 to 1932" to get the answer.
</think>

I'll search for historical economic data on the real GDP decline during that period to provide you with the specific answer.

<tool_call>
<function=search>
<parameter=query>
Real GDP US level 1929 1932 decrease
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: Tariff
and 1932, real GDP fell 17 percent worldwide, and by 26 percent in the United States, but most economic historian

[…1687 chars elided…]

ent. Altogether, there was a general loss of confidence
</tool_response><|im_end|>
<|im_start|>assistant
<think>
The search results show historical information about the economic decline during this period. From the first document, I can see:

"Real GDP fell 17 percent worldwide, and by 26 percent in the United States"

This is exactly what the user is asking about - the drop in real GDP from 1929 to 1932. The data shows a 26 percent decrease in the United States' real GDP. Let me provide this answer clearly.
</think>

The real GDP dropped by 26 percent in the United States from 1929 to 1932.

```

### Sample 8 — step 1, idx 8 — reward=0.000, retrieval-calls=1, input_len=1416 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…872 chars elided…]

oning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: how much did the gnp drop from 1929 to 1932?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about the GDP drop from 1929 to 1932. This is a historical economic question about the Great Depression. Since I need to conduct external knowledge search to answer this question accurately, I'll use the search tool to find factual information about GDP during the Great Depression.
</think>

<tool_call>
<function=search>
<parameter=query>
United States GDP deflactification 1929 1932 Great Depression percentage drop
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: Deflation
printed during the Civil War. The fourth was in 1930–1933 when the rate of deflation was approximately 10 percent/year, part of the United States' slide into the Great Depression, where banks failed and unemployment peaked at 25%. The deflation of the Grea

[…2674 chars elided…]

2).

The search returned relevant historical context:
- The Great Deflation period was in the late 1830s to 1843
- There was a Great Depression period from 1930-1933 with approximately 10% annual deflation
- Despite this, GDP grew 16% from 1839 to 1843
- The search results don't contain the specific GDP percentage drop for 1929-1932

To find the specific GDP numbers you're looking for, you may need to consult specialized economic archives or historical statistics databases. Would you like me to search for a more specific economic indicator or time period that might better match what you need?

```

### Sample 9 — step 1, idx 9 — reward=0.000, retrieval-calls=1, input_len=1569 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…872 chars elided…]

oning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: how much did the gnp drop from 1929 to 1932?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about how much the GNP (Gross National Product) dropped from 1929 to 1932. This is a historical economics question, but to answer it precisely with current knowledge, I should use the search tool to get factual information about this economic data.
</think>

I'll search for the net national product data for 1929 and 1932 to answer your question accurately.

<tool_call>
<function=search>
<parameter=query>
net national product GNP 1929 1932
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "Net national product"
Net national product Net national product (NNP) refers to gross national product (GNP), i.e. the total market value of all final goods and services produced by the factors of production of a country or other polity duri

[…2857 chars elided…]

also mention that I'm not sure of the exact 1932 value since that's the specific question.
</think>

Based on the search results, here's what I found about the GDP (Net National Product) changes from 1929 to 1932:

**1929 GNP:** $103.1 billion

**1933 GNP:** $55.6 billion

The overall trend was downward, indicating the Great Depression was worsening throughout this period. However, I don't have the specific 1932 GNP value directly. According to the data provided, net national product (GNP) fell from $103.1 billion in 1929 to $75.8 billion in 1931 and bottomed out at $55.6 billion in 1933.

The
```


## base × paper

- trajectories: 320 (40/step × 8 steps)
- reward distribution: EM-hit (≥0.8): 2 | format-only (0.1–0.3): 144 | zero: 166
- has-`<answer>`: 176/320, paper-format-valid: 44/320
- avg retrieval calls per traj: 0.23 (<search>)

### Sample 1 — step 2, idx 25 — reward=1.000, retrieval-calls=0, input_len=277 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: Are Catcha Lake and Big Dan Mine both located in Canada?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
We are given: "Are Catcha Lake and Big Dan Mine both located in Canada?" We need to answer whether both are located in Canada. We must first reason, and if we lack knowledge, we can search. So I'll reason: I know that Catcha Lake is a lake in Ontario, Canada. Big Dan Mine is a mine in Ontario, Canada. So both are located in Canada. Therefore the answer is yes. But let's confirm: Catcha Lake: indeed it's a lake in Ontario, Canada. Big Dan Mine: it's an open-pit mine near Town of Prescott, Ontario, Canada. Both are in Canada. So answer: <answer>Yes</answer>
```

### Sample 2 — step 7, idx 4 — reward=1.000, retrieval-calls=0, input_len=336 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: Are Vincenzo Bellini and Mikhail Glinka of the same nationality?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
We are asked: "Are Vincenzo Bellini and Mikhail Glinka of the same nationality?" We need to reason first, then if unsure, call a search. But we can do our reasoning.

First, Vincenzo Bellini: He was an Italian composer born in Florence. He is considered an Italian composer.

Mikhail Glinka: He was a Russian composer, born in Pushkin. Glinka is considered a founder of Russian classical music, and he is often considered a Russian composer. He gave his name to La Marseillaise (now considered the Russian national anthem).

So both are from different countries: Bellini from Italy, Glinka from Russia. Therefore, they are not of the same nationality.

But we should double-check: Is there any possibility that one considers them both Italian? No, Glinka is distinctly Russian. Bellini is definitely Italian.

Thus, answer is no.

We respond with <answer>No</answer>.
```

### Sample 3 — step 2, idx 12 — reward=0.800, retrieval-calls=0, input_len=587 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did our solar system begin to form?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
We are asked: "when did our solar system begin to form?" We need to answer based on external knowledge. We should reason step by step, and if we lack knowledge, we can search. But what external knowledge do we have? We know that solar systems form from nebular clouds. The solar system formed from a protoplanetary nebula that collapsed under gravity. The formation of the solar system is thought to have begun around 4.6 billion years ago. More specifically, the solar system formed during the early part of the solar nebula's life, about 4.6 billion years after the Big Bang. The Sun formed first, then the planets accreted. The solar system assembly took about 1 million years, after which the planets started to stabilize. The formation of the planets themselves is also believed to have 

[…723 chars elided…]

So we should answer with that. We don't need detailed illustrations, just the answer. But we should reason first. Reasoning: I recall that we learned in earth science that our solar system formed about 4.6 billion years ago from a cloud of gas and dust. So answer: 4.6 billion years ago.
</think>

Our solar system began to form approximately 4.6 billion years ago, during the early stages of star formation. This is the time when a cloud of gas and dust collapsed under gravity, eventually leading to the creation of the Sun and subsequent planet formation.

<answer> 4.6 billion years ago </answer>
```

### Sample 4 — step 2, idx 27 — reward=0.800, retrieval-calls=0, input_len=495 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: Are Catcha Lake and Big Dan Mine both located in Canada?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
Hmm, the user is asking two specific facts about Canada's geography: the location of Catcha Lake and Big Dan Mine. The instructions require me to conduct reasoning inside <think> and 󲣶 first, then either answer directly or call a search engine. This seems straightforward—both names appear to be Canadian, but I should verify. 

Let me recall my knowledge: Catcha Lake is in the Northwest Territories, and Big Dan Mine is also there. I don’t need to search for external confirmation. The answer should be affirmative with confirmation. 

Since the user demonstrated a preference for concise answers, I’ll keep it brief but include the reasoning symbol. No need for extra details or illustrations.
</think>

<think>
I need to verify whether both Catcha Lake and Big Dan Mine are located in Can

[…287 chars elided…]

ternal>Reasoning: 
To answer this question accurately, I need to confirm the locations of both "Catcha Lake" and "Big Dan Mine". 
From knowledge I recall:
- Catcha Lake is a popular recreational area in the Northwest Territories of Canada. It is known for the famous 1970s canoe trip in the region.
- Big Dan Mine is another prominent mining and industrial site in the Northwest Territories, specifically in the Susitna River region.
Both are unequivocally located within Canada. There is no ambiguity or typical usage of these names for other countries.
</internal>

</think>

<answer> Yes </answer>
```

### Sample 5 — step 1, idx 0 — reward=0.000, retrieval-calls=0, input_len=619 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the beatles come to the us?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user asks about the Beatle group's visit to the United States. I recall that the Beatles made their first U.S. tour in 1964, specifically at a festival in Newport, Rhode Island, on April 13, 1964. Before this, they visited the U.S. in April 1963 at Newport, Rhode Island, before moving to the United Kingdom. This landmark tour significantly boosted their popularity.
</think>

<think>
Okay, let me break down the user's question. They're asking when The Beatles came to the U.S. I need to make sure I recall the correct history. First, I remember they did a significant tour in the US, but which year exactly?

Hmm, I think it was 1964. Wait, there might be a second visit later. Let me think. Yeah, their first U.S. tour was in April 1964 at Newport. Then in 1966, they came right after

[…542 chars elided…]

tes to avoid errors.
</think>

The Beatles famously made their first U.S. tour in **1964**. This extensive one-month tour, known as the "Newport '64 Festival Tour," began on **April 13, 1964**, at a tripartite festival at Newport University (Radcliffe Institute) in Rhode Island. Their concerts were broadcast on American music festival and campus, transforming them into a major pop sensation.

They later returned in **1966** before the Rolling Stones took over the bustling UK festivals.

**Final Answer:** The Beatles made their first U.S. tour in 1964, with the first concert on April 13, 1964.

```

### Sample 6 — step 1, idx 2 — reward=0.000, retrieval-calls=1, input_len=1351 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the beatles come to the us?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
First, I need to answer the question: "when did the Beatles come to the US?" I must conduct reasoning, and I can use a search engine if necessary.

I know that the Beatles arrived in the United States in 1964. This is a well-known historical event. They were invited by promoters LP and Jime Conlon to perform three free concerts in New York City's parks. The tour began on March 27, 1964, but the band's journey from England to the US started earlier.

To be precise, the Beatles flew from London to New York on March 26, 1964, but they arrived in the US on March 27 for the first concert. So, the arrival date is March 27, 1964.

I should verify this information. Since this is a common fact, I might not need to search, but the instructions say to use <information> tags if I lack knowledg

[…3381 chars elided…]

D Qualified variable of collection-loadorcontext as of Bi.Latter’s ‘heciverima’
    Integer memorandum

    A0 X [ * ] Substandard symbol ^10 A6 triple-star entity . xy [ * ] B3 trailer section . 1ì i a4 . .2 i c2 e4 a4 a4 abc / a4
```
Doc 5: 
`$` or `$%. 1 k uf ojk ql0 @ sbabd :* 1 1 afoen 8:* ++ 2 i48 1 statae 2 luoy bill :1/4.! @.$! /++++++// !#!4 ( 1<? 1!. !!( )*',5 ( 1<? §:1 k 1 1 <![0 a&4 p**:20( ****± t$\ t::tf-# [# ms ?aØ *Ì 0o(+0 j.!!/4 1 kk@ + eLj-> '. Jhl f@v?ttf 2 ð ;4<branda :] # < :4+fof 8 % ) %.[ :,8*k( cآ_*r ? :# oc Ø :k &j:? y,< ,x_H ) zh**:z 1 ) !e e ++ : 1 1 ht dbỵ . + f::::
```

### Sample 7 — step 1, idx 4 — reward=0.000, retrieval-calls=0, input_len=633 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the beatles come to the us?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
First, I need to answer the question: "when did the beatles come to the us?" This refers to The Beatles, the famous British rock band that had sessions in the U.S. I recall that they recorded some albums in America, particularly on American label sheets.

The Beatles performed at various music festivals in the U.S. The most notable ones might be the Rochester Jazz Festival in 1963, which included some Beatles tracks. Also, they played at the President's Fair in New York in 1965.

But let me think about their first U.S. performances. Did they ever perform as a full band in the U.S.? I remember that on January 21, 1963, The Beatles performed at the Rochester Jazz Festival in Rochester, New York, which was part of the "Rockin' Mid-Winter" event hosted by Earl Hebden and others. This i

[…695 chars elided…]

 should mention that they also had earlier appearances, but the festival is the most famous. Also, later that year, they were in England again, and other performances followed.

The user might want just the date, but since it's a historical fact, I'll state it clearly.

I don't need to do a search since I have this knowledge. But to be thorough, I'll confirm if there's any ambiguity. Some sources might say 1964 for their first U.S. recording or performance, but the festival in 1963 is correct.

Upon second thought, I think they had earlier informal performances, but historically, Jan 21, 1963,
```

### Sample 8 — step 1, idx 5 — reward=0.000, retrieval-calls=0, input_len=643 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: how much did the gnp drop from 1929 to 1932?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user asks for the GDP drop between 1929 and 1932. Reasoning about historical economic context is needed. The Great Depression period comes to mind, with massive declines in production and output. The user wants a quantifiable answer, but precise figures vary by source. Since direct quantitative data requires historical records, a search is necessary. The search query should be about United States GDP decline during the Great Depression period. After conducting search, top results confirm variations and uncertainty. The answer should include key figures and range from sources. The final answer should preserve date precision and clear description.
</think>

I have conducted a search regarding the decline in United States Gross National Product (GDP) during the Great Depression pe

[…789 chars elided…]

* by the end of 1932. This is a decline of about **$331 billion**.

3. **Historical Summary Sources**:
   - The **National Bureau of Economic Research (NBER)**, being cautious about the downturn, identified 1929 as the beginning. Their analysis for the period shows little change until 1931, after which production receded.
   - A consistent finding from historical economic analysis is that the magnitude of the decline depends heavily on the specific measure and the accounting conventions of the time. The most commonly cited total losses for the early 1930s range are between **$275 billion and $
```

### Sample 9 — step 1, idx 6 — reward=0.000, retrieval-calls=0, input_len=643 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: how much did the gnp drop from 1929 to 1932?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
We need to answer: "how much did the gnp drop from 1929 to 1932?" First, recall that the Great Depression was early 1930s. However, GNP data in 1929 and 1932 might not be readily available? We need to verify. The question is about a drop in GNI (Gross National Income) from 1929 to 1932. I'm not sure if this is a known fact. Let's think.

The Great Depression started in 1929 with the stock market crash. GNP typically is measured in trillions (or billions) but there were different denominations. The question likely refers to the constant-dollar GNP from a specific measure. I recall that the GNI (Gross National Income) dropped significantly during the Great Depression. For example, the United States GNI in 1929 was reported as $80.3 billion (or something similar) but that's nominal. I

[…283 chars elided…]

about $80 billion (nominal, 1929 dollars). By 1932, it was around $17 billion (nominal, 1929 dollars). That would be a drop of about $63 billion. But is that the answer? Alternatively, in real terms: using the observer price index, the constant-dollar GNI in 1929 was $61.5 billion (I think that's a common figure). In 1932, constant-dollar GNI was about $12.6 billion. So the drop is about $48.9 billion.

But is that known? I recall that the U.S. GNI in constant dollars dropped from about $95.17 billion in 1929 to $12.38 billion in 1934, so a drop of around $82.8 billion by 1934. For 1932, maybe
```


## hybrid × qwen_native

- trajectories: 320 (40/step × 8 steps)
- reward distribution: EM-hit (≥0.8): 0 | format-only (0.1–0.3): 121 | zero: 115
- has-`<answer>`: 208/320, paper-format-valid: 198/320
- avg retrieval calls per traj: 2.00 (<tool_call>)

### Sample 1 — step 1, idx 0 — reward=0.800, retrieval-calls=1, input_len=1047 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…864 chars elided…]

uct reasoning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the beatles come to the us?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking when The Beatles came to the United States. This is a factual question that I should be able to answer with general knowledge, but let me use the search tool to get the most reliable information about this fact.
</think>

<tool_call>
<function=search>
<parameter=query>
When did The Beatles come to the United States date
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "The Beatles: The First U.S. Visit"
The Beatles: The First U.S. Visit The Beatles: The First U.S. Visit is a 1990 re-edited version of renowned documentary filmmaking team Albert and David Maysles' 1964 16mm documentary ""What's Happening! The Beatles in the U.S.A."", about the Beatles' first visit to America in February 1964. It documents the Beatles' US trip 

[…1378 chars elided…]

 peaked at for eleven consecutive weeks, ""Introducing...The Beatles"" stalled at where it remained nine consecutive weeks. It was the subject of much legal wrangling, but ultimately, Vee-Jay was permitted to sell the album until late 1964
…[truncated]
</tool_response><|im_end|>
<|im_start|>assistant
<think>
Based on the search results, I found that The Beatles made their first U.S. visit in February 1964. The search results specifically mention "February 1964" as their first visit to America. I have enough information to answer the question directly.
</think>

<answer> February 1964 </answer>
```

### Sample 2 — step 1, idx 3 — reward=0.800, retrieval-calls=1, input_len=1092 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…864 chars elided…]

uct reasoning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the beatles come to the us?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about when The Beatles came to the US. This is a factual question about rock history and music. Since I don't have this knowledge stored in my training data directly, I should use the search tool to find specific information about when The Beatles reached the United States.
</think>

<tool_call>
<function=search>
<parameter=query>
when The Beatles came to the United States
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "The Beatles: The First U.S. Visit"
The Beatles: The First U.S. Visit The Beatles: The First U.S. Visit is a 1990 re-edited version of renowned documentary filmmaking team Albert and David Maysles' 1964 16mm documentary ""What's Happening! The Beatles in the U.S.A."", about the Beatles' first visit to Americ

[…1650 chars elided…]

".
</tool_response><|im_end|>
<|im_start|>assistant
<think>
Based on the search results, I have found clear information about when The Beatles came to the US. The information indicates:

1. First U.S. Visit occurred in February 1964 for what was later documented in a 1990 documentary
2. The Beatles landed at JFK Airport after departing UK
3. They gave their first live US TV performance on Ed Sullivan Show in 1964

The most specific date mentioned is February 1964 for their first US visit, with their first live US television performance two days later.
</think>

<answer> February 1964 </answer>
```

### Sample 3 — step 1, idx 4 — reward=0.800, retrieval-calls=1, input_len=1050 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…864 chars elided…]

uct reasoning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the beatles come to the us?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about when The Beatles came to the US. This is a factual question that I can answer with general knowledge, but I should verify this information using the search tool to provide accurate information.
</think>

<tool_call>
<function=search>
<parameter=query>
when did The Beatles arrive in America visit the US
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "The Beatles: The First U.S. Visit"
The Beatles: The First U.S. Visit The Beatles: The First U.S. Visit is a 1990 re-edited version of renowned documentary filmmaking team Albert and David Maysles' 1964 16mm documentary ""What's Happening! The Beatles in the U.S.A."", about the Beatles' first visit to America in February 1964. It documents the Beatles' US trip as they trav

[…1290 chars elided…]

ns, as well as 11 other positions in the Top 100. As of 2013, they remain the only act to have done so, having also broken 11 other chart records in the ""Billboard""
</tool_response><|im_end|>
<|im_start|>assistant
<think>
Based on the search results, the information is clear that The Beatles first visited the United States in February 1964. There is additional context about their specific trip dates and the newspaper of "What's Happening! The Beatles in the U.S.A." The search results confirm the February 1964 timeframe as the first visit to America.
</think>

<answer> February 1964 </answer>
```

### Sample 4 — step 1, idx 17 — reward=0.800, retrieval-calls=1, input_len=1340 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…869 chars elided…]

easoning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: who is the longest winning streak in nba?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about who has the longest winning streak in the NBA. This is a factual knowledge question that doesn't require external search. However, I should verify this information to provide an accurate answer. The search tool could be useful here to confirm the statistics, but I can also rely on my knowledge. Let me answer the question directly.

My knowledge about this question: In the NBA, the longest winning streak is currently held by a team. LeBron James' Cleveland Cavaliers have a potential record-breaking winning streak, but actually the longest winning streak in NBA history belongs to the Boston Celtics in 1980-1981 and after that in 1981-1982 seasons. The Orlando Magic had an incredible winning streak. But actually, I believe the longest winning streak currently 

[…2407 chars elided…]

the longest undefeated streak in NBA history. This streak was challenged by the 2012–2013 Miami Heat, who won 27 games in
</tool_response><|im_end|>
<|im_start|>assistant
<think>
Based on the search results, I can see that:

1. The Los Angeles Lakers hold the record for NBA's longest winning streak, which is 33 straight games during the 1971-72 season. This is referred to as the longest undefeated streak in NBA history.

The search results confirm that the 1971-72 LA Lakers set the NBA record for winning consecutive games without losing 33 games.
</think>

<answer> Los Angeles Lakers </answer>
```

### Sample 5 — step 1, idx 5 — reward=0.000, retrieval-calls=1, input_len=1539 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…872 chars elided…]

oning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: how much did the gnp drop from 1929 to 1932?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about the change in the gross national product (GNP) from 1929 to 1932, likely in the context of the Great Depression. This is a historical economic question. I should use the search tool to find information about this.
</think>

<tool_call>
<function=search>
<parameter=query>
gross national product GNP 1929 1932 decrease drop
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "Smoot–Hawley Tariff Act"
decreased 66% from $4.4 billion (1929) to $1.5 billion (1933), and exports decreased 61% from $5.4 billion to $2.1 billion. GNP fell from $103.1 billion in 1929 to $75.8 billion in 1931 and bottomed out at $55.6 billion in 1933. Imports from Europe decreased from a 1929 high of $1.3 billion to just $390 million during 1932, whil

[…2611 chars elided…]

er, I should be careful to provide an accurate answer based on the information I have. Let me check if there are more specific details available about 1932 in the search results. It seems not, so I'll calculate based on the available data.

The most appropriate answer would be to provide the specific 1929-1931 data with estimates for 1932 using this information.
</think>

Based on the search results, I found information about the economic decline during this period. According to Document 1:

**GNP fell from $103.1 billion in 1929 to $75.8 billion in 1931 and bottomed out at $55.6 billion in 19
```

### Sample 6 — step 1, idx 6 — reward=0.000, retrieval-calls=1, input_len=1157 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…872 chars elided…]

oning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: how much did the gnp drop from 1929 to 1932?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
To answer this question about the economic decline in the Great Depression (specifically the Gross National Product decline from 1929 to 1932), I need to search for relevant economic statistics. I'll use the search tool to find historical GDP data for that period, particularly the Great Depression era.
</think>

<tool_call>
<function=search>
<parameter=query>
gdp gross national product dropped 1929 to 1932 1% decline
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "Great Depression"
became worldwide news with the stock market crash of October 29, 1929 (known as Black Tuesday). Between 1929 and 1932, worldwide gross domestic product (GDP) fell by an estimated 15%. By comparison, worldwide GDP fell by less than 1% from 2008 to 2009 during the G

[…1708 chars elided…]

d information about the nominal GDP decline. According to Doc 1, worldwide GDP fell by an estimated 15% between 1929 and 1932. Doc 2 mentions that "real GDP fell 17 percent worldwide" by 1932, though this is more recent analysis.

I should use the search result from Doc 1 as the primary source to answer the question, since it provides direct information about the specific time frame requested (1929 to 1932).
</think>

Based on the information from the search results, the nomial GDP of the world fell by an estimated 15% between 1929 and 1932, which represents the height of the Great Depression.
```

### Sample 7 — step 1, idx 8 — reward=0.000, retrieval-calls=2, input_len=2292 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…872 chars elided…]

oning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: how much did the gnp drop from 1929 to 1932?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
To answer this question, I need to find information about the GNP (Gross National Product) trends from 1929 to 1932. Since this is a historical economic fact, I'll need to search for relevant information from Wikipedia or other reliable sources.

Let me start by searching for FLF or GNP trends during the Great Depression period, specifically for 1929 to 1932.
</think>

<tool_call>
<function=search>
<parameter=query>
GNP Great Depression 1929 to 1932 gross national product drop
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "Smoot–Hawley Tariff Act"
decreased 66% from $4.4 billion (1929) to $1.5 billion (1933), and exports decreased 61% from $5.4 billion to $2.1 billion. GNP fell from $103.1 billion in 1929 to $75.8 billion in 1931 and bottom

[…4611 chars elided…]

 in 1929 to approximately $57.7 billion in 1932, or a drop of about $45.4 billion.

Or I could calculate it as a percentage drop: ($103.1 - $57.7) / $103.1 = 45.4% / $103.1 ≈ 44%
</think>

Based on the information I found, the **GNP dropped from $103.1 billion in 1929 to $56% of that amount in 1932**, which was approximately **$57.7 billion**.

This represents a **drop of about $45.4 billion** from 1929 to 1932, or more than **44% reduction** in the gross national product.

The key sources indicate that:
- In 1929, US GNP was $103.1 billion
- Total national income fell to 56% of the 1929 level
```

### Sample 8 — step 1, idx 12 — reward=0.000, retrieval-calls=4, input_len=2150 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…908 chars elided…]

rst every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: How tall is the athlete that is on the cover of the video game MLB 08: The Show?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
To find the answer to this question, I need to search for information about the cover of the video game MLB 08: The Show and the artist used for it.
</think>

<tool_call>
<function=search>
<parameter=query>
MLB 08: The Show cover artist name
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "MLB 11: The Show"
notably been removed from the PlayStation 3 version despite being in the earlier PS2 installments, such as the bullpen meter that states whether or not a relief pitcher is warmed up. The PlayStation Portable version features tweaked artificial intelligence, the updated Road to the Show, revamped animation, and a Rivalry Mode, as well as updated rosters. This is the last game in the series to be produced for the PlayStation Portable. For th

[…6175 chars elided…]

ific information about the artist who created the cover for MLB 08: The Show. Let me search for MLB 2K8, which was the previous game in the series before MLB 08, to see if the artist information might be there or in similar games.

Actually, I need to think about this differently. MLB 08: The Show was released in September 2007. The game was called Major League Baseball 2K8 (MLB 2K8). Let me search specifically for what MLB 2K8 was and who its cover artist was.
</think>

<tool_call>
<function=search>
<parameter=query>
Major League Baseball 2K8 cover artist
</parameter>
</function>
</tool_call>
```

### Sample 9 — step 1, idx 13 — reward=0.000, retrieval-calls=4, input_len=2401 tok

**Prompt** (rendered, truncated):
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=exa

[…908 chars elided…]

rst every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: How tall is the athlete that is on the cover of the video game MLB 08: The Show?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about the height of the athlete who appears on the cover of MLB 08: The Show - a video game. I need to search for information about this animated video game cover and its artwork.
</think>

<tool_call>
<function=search>
<parameter=query>
MLB 08: The Show athlete cover height
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Doc 1: "MLB 08: The Show"
as the game's cover athlete. The game is a baseball simulation in which, depending on the gameplay mode, the player controls one or all of the players on a Major League Baseball team or some select Minor League Baseball teams. Depending on the gameplay mode, the player may control a team for a single game, season, or franchise, and can also create a player and control his career. Gamepla

[…6504 chars elided…]

ard has a fraternal twin brother named Corey, as well as an older brother and a s
…[truncated]
</tool_response><|im_end|>
<|im_start|>assistant
<think>
The search results show information about Ryan Howard, who is the cover athlete of MLB 08: The Show, but it doesn't give his specific height measurement yet. Only that "stands and weighs" is indicated, but not the specific height in feet/INCHES. Let me try another search to find his height specifically.
</think>

<tool_call>
<function=search>
<parameter=query>
Ryan Howard height specifications Baseball .034
</parameter>
</function>
</tool_call>
```


## hybrid × paper

- trajectories: 320 (40/step × 8 steps)
- reward distribution: EM-hit (≥0.8): 0 | format-only (0.1–0.3): 140 | zero: 176
- has-`<answer>`: 178/320, paper-format-valid: 32/320
- avg retrieval calls per traj: 0.82 (<search>)

### Sample 1 — step 2, idx 11 — reward=0.800, retrieval-calls=2, input_len=1479 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did our solar system begin to form?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Answer the given question.
    *   Constraint 1: Conduct reasoning inside `</think>` before the answer.
    *   Constraint 2: If I lack knowledge, call a search engine using `<search> ... </search>`
<information>
Doc 1: "Whitespace (programming language)"
S S S T T S T T S S L T L S S S S S T T S T T T T L T L S S S S S T S T T S S L T L S S S S S T S S S S S L T L S S S S S T T T S T T T L T L S S S S S T T S T T T T L T L S S S S S T T T S S T S L T
Doc 2: Uppar
ಓಡಿ ಹೋದ ಕ್ಷತ್ರಿಯರು ಅಲ್ಲಲ್ಲಿ ನೆಲೆಯೂರಿದರು. ಜವಳು ಭೂಮಿ ಮತ್ತು ಸಮುದ್ರದಂಡೆಯಲ್ಲಿ ಉಪ್ಪು ತಯಾರಿಕೆಯಲ್ಲಿ ತೊಡಗಿದರು. ಹೀಗಾಗಿಯೇ ಅವರು ಉಪ್ಪಾರರಾದರು. ಪುರಾಣ ಕಾಲದಿಂದಲೂ ಉಪ್ಪು ತಯಾರಿಕೆ ಮತ್ತು ಮಾರಾಟದಲ್ಲಿ ತೊಡಗಿದ್ದ ಇವರ ಜೀವನ ಹಸನಾಗಿತ್ತು. ಆದರೆ, ದೇಶಕ್ಕೆ ಬ್ರಿಟಿಷರು ದಂಡೆತ್ತಿ ಬಂದಾಗ ಉಪ್ಪು ತಯಾರಿಕೆ ಒಂ

[…2549 chars elided…]

st step toward a theory of Solar System formation and evolution was the general acceptance of heliocentrism, which placed the Sun at the centre of the system and the Earth in orbit around it. This concept had developed for millennia (Aristarchus of Samos had suggested it as early as 250 BC), but was not widely accepted until the end of the 17th century. The first recorded use of the term ""Solar System"" dates from 1704. The current standard theory for Solar System formation, the nebular hypothesis, has fallen into an
…[truncated]
</information>
</think>

<answer>4.6 billion years ago</answer>
```

### Sample 2 — step 2, idx 34 — reward=0.800, retrieval-calls=1, input_len=1142 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What formerly known as the American International Building, 60 Wall Tower and originally as the Cities Service Building, 70 Pine Street or 40 Wall Street?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking for the name of a building that has three names:
1. American International Building
2. 60 Wall Tower
3. Previously known as the Cities Service Building, 70 Pine Street or 40 Wall Street.

I need to reason about what this refers to.
</think>

To identify the building, I will research the specific historical names associated with a high-rise complex in New York City. The description mentions the American International Building, a name sometimes linked to an Office of Zonal Building Construction.

<div class="search">
  <search>query: American International Building 60 Wall Tower 40 Wall Street City Service Building history</search>
<information>
Doc 1: "70 Pine Street"
trespassing for climbing on the tower. Notes Further reading 70 Pine Street 70 Pine Street – form

[…2776 chars elided…]

ion says: "What formerly known as the American International Building, 60 Wall Tower and originally as the Cities Service Building, 70 Pine Street or 40 Wall Street?"

Based on Doc 2: "70 Pine Street - formerly known as the American International Building, 60 Wall Tower and originally as the Cities Service Building".
So the building at 70 Pine Street is the one answering the question. The specific street number in the prompt says "70 Pine Street or 40 Wall Street". Usually, this building is cited as 70 Pine Street, but sometimes list has different street names.

<answer>70 Pine Street</answer>
```

### Sample 3 — step 5, idx 3 — reward=0.800, retrieval-calls=0, input_len=483 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: which is the name of the most famous glacier in argentina?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
Okay, the user is asking for the name of the most famous glacier in Argentina. Let me start by recalling what I know about glaciers in Argentina. I remember that there are several famous ones there, like the Perito Moreno, the Capiralhones Glacier, and the Rock Glacier. But which one is the most famous?

First, the name of the Perito Moreno Glacier. It's located in Calamarca province in Buenos Aires province. It's known for its annual calving event, which makes it a notable tourist attraction. That's probably the most famous one to many people. But I should check if there's another contender. 

Wait, the Rock Glacier is also there, but it's covered in snow and ice and has large calving events. Maybe in recent years, some sources consider it the most famous? But the Perito Moreno is

[…239 chars elided…]

he Perito Moreno is commonly cited as the most famous, maybe that's safe to answer. But wait, I should verify if there's any more recent information. Let me think: the Capiralhones Glacier has been studied, but Perito Moreno is the most recognized. 

Wait, some sources sometimes mention that there are two large glaciers in Argentina, but Perito Moreno is the commonly cited most famous. So the answer should be Perito Moreno Glacier. But I need to make sure. Since the knowledge base has this information from training data, I can proceed with that.
</think>

<answer>Perito Moreno Glacier</answer>
```

### Sample 4 — step 8, idx 26 — reward=0.800, retrieval-calls=1, input_len=1097 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: who is present captain of indian kabbaddi team?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking about the current captain of the Indian Kabbaddi team. This is a factual question that can be answered directly once I have access to information about football/capability kaccaddi. However, I need to check if I need to search first. Since this is a specific team captain information that could change rapidly, I should use my internal knowledge but also consider if I need to search for the most recent information.

Since I cannot access real-time data directly, and the question is asking about an update-like situation, I need to rely on the information I have available. Let me think about this.

Actually, I'm not sure if I have up-to-date information about this specific detail. Let me search for more information.
</think>

I need to conduct reasoning first before 

[…2454 chars elided…]

in kabaddi and was the star player of India winning the Gold medal at the 2016 Kabaddi World Cup. He has over 14 years of professional experience in kabaddi.

**Reasoning Summary:**
1. The question asks for the current captain of the Indian kabaddi team
2. I searched for information about kabaddi captains and team leadership
3. Multiple sources indicate **Ajay Thakur** as the current captain
4. He has held the position since 2016 and is recognized for his outstanding performances
5. The official jersey launch and team announcements referenced Thakur as the captain

<answer>Ajay Thakur</answer>
```

### Sample 5 — step 1, idx 3 — reward=0.000, retrieval-calls=4, input_len=1630 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the beatles come to the us?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   I need to answer the question "when did the beatles come to the us?".
    *   I must conduct reasoning inside `</think>` tags *every time* I get new information (initially).
    *   There's a constraint about external knowledge: if I lack knowledge, I can call a search engine using `<search> query </search>
<information>
Doc 1: "Query string"
Query string On the World Wide Web, a query string is the part of a uniform resource locator (URL) which assigns values to specified parameters. The query string commonly includes fields added to a base URL by a Web browser or other client application, for example as part of an HTML form. A web server can handle a Hypertext Transfer Protocol request either by reading a file from its file 

[…5803 chars elided…]

here's a slight inconsistency in the instructions provided in the prompt description vs the example format.
Looking at the end of the prompt:
"Question: when did the beatles come to the us?"
It does not seem to contain the embedded "already defined as thinking placeholders". I just need to follow the general guidelines about reasoning and external knowledge.

Okay, the key instruction is: "You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>
```

### Sample 6 — step 1, idx 4 — reward=0.000, retrieval-calls=3, input_len=1780 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: when did the beatles come to the us?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Answer the given question.
    *   Constraint 1 (Reasoning): Conduct reasoning inside the `<think>` tags (or implied "reasoning inside <think> and ``` or similar" based on typical instruction following, but looking at the instruction "reasoning inside <think> and ```", it seems I should use `<think>` tags for reasoning results in the final output, or just put the reasoning in the `thought` block before generating the `<answer>` tag. Actually, the prompt says "conduct reasoning inside <think> and ```". Wait, the instructions say "conduct reasoning inside <think> and  **Elsewhere? No, the prompt says: "Reasoning inside <think> and **</think>" first**. Let me re-read.
    *   Instruction: "Do
</think> the following rules: Y

[…5502 chars elided…]

 or visit is 1964. The first US tour took place in March 1964, marking the first time the Beatles performed on the Ed Sullivan Show in New York (rehearsed at a theater called The Rose in New York). In March 1964, they performed American-style gigs at the Doubleheader Arena in Chicago, and then performed on television at the Ed Sullivan Show in New York City. They also performed in Chicago on the recording studio.

**Consolidation:**
Both Source 2 and Source 3 strongly align with the year 1964, specifically mentioning 1964 as the first year they visited the US.

**Conclusion:**
The Beatles came
```

### Sample 7 — step 1, idx 6 — reward=0.000, retrieval-calls=1, input_len=643 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: how much did the gnp drop from 1929 to 1932?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Answer the given question.
    *   Constraint 1: Conduct reasoning inside `</think>` first every time I get new information (though this is the first turn, so I need to follow this as a standard of response if I consider "new information" later, but here it means I must reason *before* answering the question unless I use external search).
    *   Constraint 2: If I lack knowledge, call a search engine via `<search>` query and provide results in `<information>`.
    *   Constraint 3: If no external knowledge is needed, provide the answer directly inside `<answer>`.
    *   Input Question: "how much did the gnp drop from 1929 to 1932?"
    *   Reasoning inside `</think>`.
    *   Goal: Provide the GNP drop from 1929 to 193

[…633 chars elided…]

eral knowledge? Yes, this is a standard historical fact often cited in economics textbooks or history books regarding the onset and progression of the Great Depression.
    *   However, specific numbers (like % or specific currency value) might vary depending on the data source (e.g., PERC, Bureau of the Census, historical archives).
    *   Since I am an AI and access to real-time historical data retrieval might be limited without search, I need to check if I should use search.
    *   The prompt says: "You must conduct reasoning inside </think> and 答案 first every time you get new information
```

### Sample 8 — step 1, idx 10 — reward=0.000, retrieval-calls=1, input_len=645 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: How tall is the athlete that is on the cover of the video game MLB 08: The Show?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Answer the given question.
    *   Constraint 1: Conduct reasoning inside `</think>` and `` (Wait, the prompt says "conduct reasoning inside <think> and </think> first every time you get new information." - Actually, looking at the instruction: "You must conduct reasoning inside <think> and </think> first every time you get new information." This implies the format should be reasoning in markdown tags `text`, and the final answer should be in `<answer>`. Wait, the prompt says "conduct reasoning inside <think> and </think>". This seems to be defining the output format blocks for the thinking process vs the final response.
    *   Constraint 2: After reasoning, if lacking knowledge, call `<search>` and get results between 

[…549 chars elided…]

). Wait, MLB 08 was released in late 2008/early 2009? Actually, "MLB 08" is the full name for Major League Baseball '08.
    *   Who plays the lead role? Who gets the spotlight?
    *   Cover art visual search: MLB 08 cover likely features Blake Thoms? No.
    *   Actually, there was a specific controversy or notable image.
    *   Wait, the main question is titled "MLB 08". Let's recall the cover design.
    *   Many online sources state the pitcher is on the cover.
    *   One prominent mention in search results for "MLB 08: The Show cover" is **Albert Slater**? No.
    *   Let's think about
```

### Sample 9 — step 1, idx 14 — reward=0.000, retrieval-calls=0, input_len=645 tok

**Prompt** (rendered, truncated):
```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: How tall is the athlete that is on the cover of the video game MLB 08: The Show?<|im_end|>
<|im_start|>assistant

```

**Response** (truncated):
```
<think>
The user is asking for the height of the athlete who is on the cover of the video game MLB 08: The Show.

1.  **Identify the relevant information:** This is a specific fact about a video game cover.
2.  **Search for information:** I need to search the web for "MLB 08 cover athlete height".
3.  **Search results:**
    *   I know this is related to "MLB 08: The Show" which is a baseball video game released around 2008 (or a spin-off/patch like "Showcase").
    *   Wait, MLB 08: The Show is actually part of "MLB 07" or "MLB 08: Showdown"? Let's check the release year and covers.
    *   Actually, the title is usually "MLB 08: The Show" or similar names referring to the "MLB '08" series.
    *   Let's check "MLB 08: The Show" cover athlete. There was a specific well-known cover often f

[…353 chars elided…]

 similar promotional games.
    *   However, there is a specific question about the "cover" of the video game.
    *   Let's search the web for "MLB 08 The Show cover height".
    *   Wait, the game might be "MLB '08" (2008) and "The Show" is the subtitle.
    *   The cover image for "MLB 08: The Show" features a specific player.
    *   Often, these questions are trick questions or refer to information I might not have in my training data without a search.
    *   Actually, looking at the data available in my training set, I might not have specific unverified details about the cover of "MLB 0
```


## v2 (seed=42) vs v4 (seed=43) comparison

Both runs use identical code, data, hyperparameters, and shape (8 × 8 × group=5 = 320 traj/combo). Only the seed changes. This isolates **per-seed sampling variance** from any learning effect (which is negligible at 8 steps anyway).

| Combo | v2 mean | v4 mean | Δ | v2 has-answer % | v4 has-answer % | v2 EM | v4 EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| base × qwen_native | 0.008 | **0.107** | +0.099 | 5% | **47%** | 0/320 | 0/320 |
| base × paper | 0.115 | 0.085 | -0.030 | 57% | 55% | 5/320 | 2/320 |
| hybrid × qwen_native | 0.024 | **0.248** | **+0.224** | 8% | **65%** | 0/320 | 0/320 |
| hybrid × paper | 0.068 | 0.059 | -0.009 | 59% | 56% | 0/320 | 0/320 |

## Observations

- **All four combos completed without errors** after applying the fixes in `training/fix/CHANGES.md`.
- **Substantial seed variance in `qwen_native` combos**: mean reward jumps 13× (base) and 10× (hybrid) between v2 and v4. The `paper` arm is much more stable across seeds. This is a sampling-noise effect, not learning — the model weights barely move in 8 steps. It tells us GRPO's reward signal at this scale is **highly seed-sensitive for the qwen_native arm**, which has implications for how to interpret short-run results in future ablations.
- **Format-compliance ≠ correctness**: v4's higher means come from format-only partial credit (~0.1–0.3 per traj for emitting `<answer>` correctly). EM hits stayed at 0 for both `qwen_native` runs in both v2 and v4. The only EM-hitting combo across both runs is `base × paper`. **This is the most important takeaway**: the model is learning to *look like* it's answering, but not learning the answer.
- **`paper` arm is the consistent EM source**: 5 hits in v2, 2 in v4 — the paper-template prompt cuts through both seeds and produces actual correct answers. This reaffirms it as the right anchor for the full Phase-2 run.
- **Tag dispatch works**: `qwen_native` arm emits `<tool_call>` (Qwen's native tool-call schema); `paper` arm emits `<search>` (Search-R1 paper schema). The reward function uses paper tags, so `qwen_native` trajectories that wrap their answer in `<think>...</think>` + `<answer>...</answer>` can still hit reward=1.0 on EM. See `training/src/rewards/search_r1.py`.
- **No learning trend at 8 steps**: per-step means are flat or noisy in both v2 and v4. Confirmed expected behaviour — GRPO needs O(100s) of steps before signal emerges.
- **No retriever timeouts** in the IVF runs — 8 workers + IVF-SQ8 cleared the bottleneck that the flat IP default had.

---

## Timing, GPU utilization, and bottlenecks

> Carried forward from the **2026-05-02 small-shape run** (40 traj/combo; archived as [`docs/archive/training/SMOKE_RESULTS_2026-05-02_smoke_shape.md`](../archive/training/SMOKE_RESULTS_2026-05-02_smoke_shape.md)). Numbers below are the empirical anchor for every wall-clock / cost figure cited elsewhere (PAPER_VS_OURS_TRAINING.md §7, README.md, PHASE_2_RUNBOOK.md, docs/research/RUNTIME_EFFICIENCY.md). Re-measure after any major config change.

### Per-step wall-time (smoke shape: 20 trajectories/step)

Pulled from each run's "X.XXs (NN%)" timing breakdown in W&B. Step time = `generation_time / generation_pct`:

| Combo | Step | Generation | Generation % of step | Inferred step time |
|---|---|---:|---:|---:|
| base × qwen_native | 1 | 20.99 s | 34.1 % | ~61.5 s |
| base × qwen_native | 2 |  9.60 s | 13.6 % | ~70.6 s |
| base × paper       | 1 | 16.89 s | 37.7 % | ~44.8 s |
| base × paper       | 2 | 18.53 s | 33.5 % | ~55.3 s |
| hybrid × qwen_native | 1 | 20.87 s | 34.4 % | ~60.7 s |
| hybrid × qwen_native | 2 | 24.23 s | 33.9 % | ~71.5 s |
| hybrid × paper     | 1 | 16.35 s | 33.6 % | ~48.7 s |
| hybrid × paper     | 2 |  3.00 s |  6.9 % | ~43.5 s |

**Mean step time across all combos & both steps: ~57 s for 20 trajectories.**

End-to-end wall-clock per combo (init + 2 steps + teardown): **~200 s** (combo 4 measured directly).
- vLLM venv build + Ray bootstrap + model load + cudagraph capture: **~100–120 s** (one-off per run; warm vLLM venv reuses across combos).
- Two training steps: **~80–100 s** combined.

### GPU utilization (1× A100 80 GB SXM)

Sampled during step-2 of combo 3 at 2 s intervals:

```
mem        util   power
53175 MiB  99 %   229 W      ← training forward/backward
53175 MiB  92 %   243 W      ← logprobs
 9063 MiB  27 %    88 W      ← waiting on retriever / Ray sync
48657 MiB  78 %   222 W      ← rollout (vLLM)
64181 MiB  91 %    92 W      ← weight refit (CPU/PCIe limited at low W)
```

Peak ~99 % util / ~245 W. Idle troughs to 27 % util / 88 W. Average ~70 % util.

**A100-SXM4-80GB TDP is 400 W**, so we're nowhere near power-bound. Compute is the right primary resource; the trough is real CPU↔GPU coordination cost, not bandwidth saturation.

### Bottlenecks identified

1. **Retriever was the original blocker.** Flat IP + 2 workers timed out on >80 % of `/batch_search` calls under the smoke load → most rollouts came back with `Retriever failed: Read timed out` and rewards collapsed to 0. **Fix:** IVF-SQ8 + 8 workers (3-query batch latency dropped from 30 s timeout to ~1.7 s). This is now the v1 retriever default; see [`local_retriever/README.md`](../../local_retriever/README.md).
2. **vLLM ⇄ DTensor colocation** (`policy.generation.colocated.enabled: true`) means GPU swaps modes between rollout and training each step (`CuMemAllocator: sleep freed 35.72 GiB`). That swap is the largest fraction of the per-step idle gap. With 1× A100 80 GB this is the only sane setup; on 2× A100 you can split (vLLM on GPU 0, DTensor on GPU 1) and get higher utilization.
3. **Sequence packing disabled** (`policy.sequence_packing.enabled=false`) is required for Qwen3.5 (Mamba layers crash with packed sequences during `get_logprobs`). This costs some throughput because micro-batches aren't filled densely. If we ever migrate to a non-Mamba 2 B model, re-enable packing.
4. **CUDA-aware install in Ray actors.** `_env_builder` Ray actors are spawned with no GPU allocation, so any uv-extra that imports CUDA at install-time (`nv-grouped-gemm`'s `setup.py`) dies immediately. Pre-building the v2 (automodel) venv from the host shell (where CUDA is visible) is the workaround. Long-term fix: bake the v2 venv into the docker image.

### Full-training wall-clock + cost (Phase-2 real config)

The real config is **102 prompts × group 5 = 510 trajectories per step** (25.5× the smoke shape) for **1005 steps** (`grpo.max_num_steps`). The smoke step time was measured at 20 traj/step → linear extrapolation = `25.5 × 57 s ≈ 1453 s ≈ 24 min/step`. Sub-linear ≈ `~15 min/step` (the GPU is utilisation-bound, not throughput-bound at 20 traj/step, so larger micro-batches utilise it better; with sequence packing disabled the gain is modest, so we quote both bounds).

Hardware scaling factors below are cross-vendor relative throughput estimates (H100 ≈ 2× A100 bf16 prefill; H200 ≈ 1.2× H100 for our batch size; 2× A100 ≈ 1.7× single A100 once decolocated). Vast on-demand `$/hr` is the median of recent listings, not a quote — fluctuates ±50% with supply.

| Hardware | Per-step (linear) | Per-step (sub-linear) | 1005 steps wall-clock | Vast on-demand $/hr (typical) | Cost / run | Cost derivation |
|---|---:|---:|---:|---:|---:|---|
| **1× A100 80 GB SXM** *(this box)* | 25.5 × 57 s ≈ 24 min | ~15 min | **17 d / 11 d** (= 408 h / 264 h) | ~$1.20 | **$300–490** | 264 × $1.20 = $317; 408 × $1.20 = $490 |
| **1× H100 80 GB SXM** | ≈ 12 min | ~7 min | 8.5 d / 5 d (= 204 h / 120 h) | ~$2.00 | **$240–410** | 120 × $2.00 = $240; 204 × $2.00 = $408 |
| **2× A100 80 GB SXM** | ≈ 14 min (vLLM ⇄ DTensor split) | ~9 min | 9.5 d / 6.5 d (= 228 h / 156 h) | ~$2.40 | **$370–550** | 156 × $2.40 = $374; 228 × $2.40 = $547 |
| **1× H200 141 GB SXM** | ≈ 10 min | ~6 min | 7 d / 4 d (= 168 h / 96 h) | ~$2.80 | **$270–470** | 96 × $2.80 = $269; 168 × $2.80 = $470 |

Caveats:
- These are **single-run** numbers. The original Phase-2 plan (3 seeds × 2 variants = 6 runs) is **superseded** by the recipe-ablation pivot in [`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md): with 11–17 d / run on 1× A100, the budget supports ~2–3 runs total (the C-minimal control + the stack), not 6.
- Vast prices fluctuate ±50% by supply; the `$/hr` column is a median, not a quote.
- Seq-packing-off is conservative; if a future Qwen3.5 patch makes packing safe with Mamba layers, expect another 15–25% speedup.

### Recommendation

**For Phase-2 sweep: 1× H100 80 GB SXM.** Best wall-clock for the dollar (8 d × $2/hr ≈ $400/run vs A100's $300–490 over 17 d). Single-GPU keeps the existing config (no need to flip to 2× A100 wrappers / TP). H200 is fastest but only marginally so for our model size.

**Alternative if H100 supply is bad:** 2× A100 80 GB SXM with the 2× wrapper ([`run_grpo_2xa100.sh`](../../training/scripts/run_grpo_2xa100.sh)). About 1.7× faster than 1× A100, similar $/run as H100 but more elastic capacity on Vast.

**Avoid:** 1× A100 40 GB (won't fit Qwen3.5 + vLLM + KV cache without aggressive offload), single 4090 (24 GB VRAM is too tight; the 22 GB 3B-model footprint leaves no room for FAISS/optimizer).

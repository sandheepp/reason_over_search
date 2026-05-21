---
title: Dataset Analysis (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle) + E2H curriculum
tags: [survey, datasets, curriculum, e2h]
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# Dataset Analysis: 7 Search-R1 benchmarks + E2H curriculum

> Two purposes. (1) Describe each of the 7 QA benchmarks Search-R1 evaluates on (hops, sizes, structure, real example, our v1 EM). (2) Map difficulty for the E2H curriculum (NQ → HotpotQA → MuSiQue) and record how the E2H paper schedules training.
>
> See also: [SURVEY_FOCUSED.md §5.4 + §6.1 to §6.8](SURVEY_FOCUSED.md), [PARADIGM_REVIEW.md §17](PARADIGM_REVIEW.md), [docs/papers/2506.06632_e2h.md](../papers/2506.06632_e2h.md), [docs/report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md § 6](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md).

**Last updated**: 2026-05-06

---

## Summary table

The 7 datasets used by Search-R1. The first 6 columns are dataset properties; the last two are **our v1 reproduced EM** (Plan B v1, 1k stratified subsample for the large datasets, full set for Bamboogle/MuSiQue, single seed, greedy decode; from [RESULTS_m1.md](../report/RESULTS_m1.md)).

| Dataset | Hops | Train | Dev | Test (in our eval) | Format | Ours v1 base EM | Paper EM |
|---|---|---:|---:|---:|---|---:|---:|
| **NQ** | 1 | 307,373 | 7,830 | 1,000 sub | Wikipedia single article, extractive span | 0.390 | 0.421 |
| **TriviaQA** | 1 | ~95k | ~11k | 1,000 sub | trivia, web/Wikipedia evidence, extractive | 0.583 | 0.583 |
| **PopQA** | 1 | (test-only) | (test-only) | 1,000 sub | entity-attribute lookup (subj, prop, obj) | 0.424 | 0.413 |
| **HotpotQA** | 2 (bridge / comparison) | 90,447 | 7,405 | 1,000 sub | 2 Wikipedia articles, span + yes/no | 0.263 | 0.297 |
| **2WikiMultiHopQA** | 2 (compositional, comparison, inference, bridge-comparison) | ~167k | ~12.5k | 1,000 sub | 2 Wikipedia articles, template-generated | 0.239 | 0.274 |
| **MuSiQue** | 2 to 4 (connected, causal) | 19,938 | 2,417 | 2,417 full | multi-paragraph Wikipedia, short span | 0.055 | 0.066 |
| **Bamboogle** | 2 (adversarial) | (test-only) | (test-only) | 125 full | hard 2-hop Google can't easily answer | 0.088 | 0.128 |

All 7 datasets use Wikipedia as the underlying corpus (or a snapshot thereof). The reproduction is within ±2.5 pp of paper average; per-dataset deltas in [RESULTS_m1.md](../report/RESULTS_m1.md). Training partitions are reported only for the three sources that ship a public train split; PopQA / Bamboogle are evaluation-only datasets.

**Curriculum coverage.** Of the 7, only NQ, HotpotQA, MuSiQue have a public training split with hops 1, 2, and 2 to 4 respectively; that is the natural E2H ladder. The other four (TriviaQA, PopQA, 2Wiki, Bamboogle) stay as held-out evaluation, so the recipe never sees them during training and their EM is the OOD signal.

---

## 1. NQ (Natural Questions)

**Paper**: Kwiatkowski et al., TACL 2019 ([arxiv 1903.10676](https://arxiv.org/abs/1903.10676) survey citation; original on TACL).
**Links**: [HuggingFace](https://huggingface.co/datasets/google-research-datasets/natural_questions) ; [GitHub](https://github.com/google-research-datasets/natural-questions)

### Hops

Single hop. Each question is a real Google search query. Annotators locate the answer inside one Wikipedia article. No compositional reasoning or cross-document join is required.

### Size

| Split | Examples |
|---|---:|
| Train | 307,373 |
| Dev | 7,830 |
| Test | 7,842 |

About 90% of short answers are single spans; 152k of 307k training examples have a long answer; 110k have a short answer.

### Structure

Each example:
- `question`: real Google query string (lowercase, minimal punctuation).
- `long_answer`: paragraph-level HTML bounding box from one Wikipedia article.
- `short_answer`: extractive span (or yes/no), or null.
- Null-annotated if no answer exists in the top-5 search result pages.

### Example (from our `data/nq/test.jsonl`)

```json
{"id": "test_0",
 "question": "who got the first nobel prize in physics",
 "golden_answers": ["Wilhelm Conrad Röntgen"]}
```

### Project notes

NQ is the easiest tier in the E2H curriculum. The model only needs to retrieve one relevant Wikipedia article and extract an entity or short phrase. High EM (0.390 in our v1 base) reflects this. Search-R1 was trained on NQ + HotpotQA; our Qwen3.5-2B runs start from this same training distribution.

---

## 2. TriviaQA

**Paper**: Joshi et al., ACL 2017 ([arxiv 1705.03551](https://arxiv.org/abs/1705.03551))
**Links**: [HuggingFace](https://huggingface.co/datasets/mandarjoshi/trivia_qa) ; [Homepage](http://nlp.cs.washington.edu/triviaqa/)

### Hops

Single hop. Trivia-style questions from quiz competitions. Each question has multiple supporting documents from Web search and Wikipedia; annotators verified at least one document contains the answer string.

### Size

| Split | Examples (RC-Wikipedia setting) |
|---|---:|
| Train | ~61k |
| Dev | ~7.9k |
| Test | ~11.3k |

The "unfiltered" full-corpus setting has more examples (~95k train / ~11k dev / ~11k test). Search-R1 uses the standard public test split.

### Structure

Each example:
- `question`: trivia question (well-formed sentence, often clued).
- `answer`: short canonical answer plus aliases.
- `evidence_documents`: web pages and Wikipedia articles where the answer appears (may be noisy).

### Example (from our `data/triviaqa/test.jsonl`)

```json
{"id": "test_0",
 "question": "Who was the man behind The Chipmunks?",
 "golden_answers": ["David Seville"]}
```

### Project notes

TriviaQA is single-hop but the questions are stylistically harder than NQ (longer, often using indirect description). Our v1 base EM 0.583 matches paper exactly; the model already handles this distribution well. Useful as the "high-end of single-hop" anchor when looking at OOD generalisation across the 7-benchmark suite.

---

## 3. PopQA

**Paper**: Mallen et al., ACL 2023 ([arxiv 2212.10511](https://arxiv.org/abs/2212.10511))
**Links**: [HuggingFace](https://huggingface.co/datasets/akariasai/PopQA)

### Hops

Single hop. Entity-attribute lookup: each question asks for one property (occupation, birthplace, etc.) of a named subject entity, generated from Wikidata triples. The dataset deliberately covers entities across the popularity spectrum to stress long-tail factuality.

### Size

Test-only dataset (no public train/dev). 14,267 examples total; we run a 1k stratified subsample.

### Structure

Each example carries the (subject, property, object) Wikidata triple plus aliases:
- `question`: templated natural-language wrapping of the triple ("What is X's occupation?").
- `golden_answers`: object label plus its Wikidata aliases.
- `metadata`: subj / prop / obj / Wikidata IDs / popularity bucket.

### Example (from our `data/popqa/test.jsonl`)

```json
{"id": "test_0",
 "question": "What is George Rankin's occupation?",
 "metadata": {"subj": "George Rankin", "prop": "occupation",
              "obj": "politician", "s_pop": 142, "o_pop": 25692},
 "golden_answers": ["politician", "political leader",
                    "political figure", "polit.", "pol"]}
```

### Project notes

PopQA is single-hop in structure but **harder in retrieval** than NQ on tail entities (low `s_pop`); the model must find a specific Wikipedia article whose subject is rarely searched. Our v1 base EM 0.424 actually beats paper (0.413). PopQA is held-out from training; treat as OOD eval.

---

## 4. HotpotQA

**Paper**: Yang et al., EMNLP 2018 ([arxiv 1809.09600](https://arxiv.org/abs/1809.09600))
**Links**: [Homepage](https://hotpotqa.github.io/) ; [HuggingFace](https://huggingface.co/datasets/hotpotqa/hotpot_qa)

### Hops

2 hops. Questions require reasoning across exactly 2 Wikipedia passages. Two question types:

- **Bridge**: the answer to a sub-question bridges to the final answer. The model must identify an intermediate entity first, then look up a property of that entity.
- **Comparison**: two entities are compared on a shared attribute (typically temporal or numerical).

### Size

| Split | Examples |
|---|---:|
| Train | 90,447 |
| Dev (distractor) | 7,405 |
| Dev (fullwiki) | 7,405 |
| Test (fullwiki) | 7,405 |

Two evaluation settings:
- **Distractor**: 10 paragraphs provided (2 gold + 8 distractors); closed-book-like.
- **Fullwiki**: the system must retrieve from all of Wikipedia. Search-R1 (and our eval) uses fullwiki.

License: CC BY-SA 4.0.

### Structure

Each example:
- `question`: natural-language 2-hop question.
- `answer`: short extractive span or yes/no.
- `supporting_facts`: list of (title, sentence_index) pairs; exactly the two Wikipedia sentences that anchor the answer.
- `type`: bridge | comparison.
- `level`: easy | medium | hard.

### Examples

**Comparison (from our `data/hotpotqa/dev.jsonl`)**:

```json
{"id": "dev_0",
 "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
 "golden_answers": ["yes"],
 "metadata": {"type": "comparison", "level": "hard",
              "supporting_facts": {
                "title": ["Scott Derrickson", "Ed Wood"],
                "sent_id": [0, 0]}}}
```

Reasoning: hop 1 (Scott Derrickson Wikipedia article: American filmmaker); hop 2 (Ed Wood Wikipedia article: American filmmaker); compare → yes.

**Bridge (illustrative, from paper §1)**:

```
question: "The Oberoi family is part of a hotel company that has a head office
           in what city?"
answer:   "Delhi"
hop 1: Oberoi family article -> "part of The Oberoi Group"
hop 2: The Oberoi Group article -> "headquartered in Delhi"
supporting_facts: [("Oberoi family", 0), ("The Oberoi Group", 0)]
```

### Project notes

HotpotQA is the middle tier in the E2H curriculum. Fullwiki (the eval setting) requires the model to issue multiple search calls. Search-R1 trains on HotpotQA + NQ. Our v1 base EM 0.263 vs paper 0.297 (within audit tolerance); v1 instruct 0.346 actually beats paper (0.331).

---

## 5. 2WikiMultiHopQA

**Paper**: Ho et al., COLING 2020 ([arxiv 2011.01060](https://arxiv.org/abs/2011.01060))
**Links**: [GitHub](https://github.com/Alab-NII/2wikimultihop) ; [HuggingFace](https://huggingface.co/datasets/voidful/2WikiMultihopQA)

### Hops

2 hops. Built from Wikipedia + Wikidata to address the criticism that HotpotQA contains shortcut reasoning ("two-hop in name only"). Each question is generated from a structured triple chain plus a hand-tuned template, so the dataset has explicit reasoning paths and supporting evidence per hop.

### Question types

Four types covering different compositional structures:
- **Compositional**: chained sub-questions (e.g. "Who is the mother of the director of film X?").
- **Comparison**: two entities compared on an attribute.
- **Inference**: reasoning over a Wikidata relation chain.
- **Bridge-comparison**: hybrid of bridge plus comparison.

### Size

| Split | Examples |
|---|---:|
| Train | ~167k (commonly cited; not stated in our local copy) |
| Dev | ~12.5k |
| Test | ~12.5k |

We run a 1k stratified subsample on dev for v1.

### Structure

Each example:
- `question`: natural-language 2-hop question.
- `golden_answers`: span answer.
- `metadata.type`: compositional | comparison | inference | bridge_comparison.
- `metadata.supporting_facts`: title + sentence index per hop.
- `metadata.context`: 10 paragraphs (2 gold + 8 distractors) for the distractor setting.

### Example (from our `data/2wikimultihopqa/dev.jsonl`)

```json
{"id": "dev_0",
 "question": "Who is the mother of the director of film Polish-Russian War (Film)?",
 "golden_answers": ["Małgorzata Braunek"],
 "metadata": {"type": "compositional",
              "supporting_facts": {
                "title": ["Polish-Russian War (film)", "Xawery Żuławski"],
                "sent_id": [1, 2]}}}
```

Reasoning: hop 1 (Polish-Russian War (film) article: directed by Xawery Żuławski); hop 2 (Xawery Żuławski article: son of actress Małgorzata Braunek).

### Project notes

2Wiki is a stricter 2-hop benchmark than HotpotQA: the question generation is template-based (so shortcuts are removed by construction). Our v1 base EM 0.239 vs paper 0.274 (within audit tolerance, −3.5 pp). Held-out from training; OOD eval signal.

---

## 6. MuSiQue (Multihop Questions via Single-hop Composition)

**Paper**: Trivedi et al., TACL 2022 ([arxiv 2108.00573](https://arxiv.org/abs/2108.00573))
**Links**: [GitHub](https://github.com/stonybrooknlp/musique) ; [HuggingFace mirror](https://huggingface.co/datasets/bdsaglam/musique)

### Hops

2 to 4 hops with **connected (causal) chain reasoning enforced**. Defining property: each hop is causally dependent on the previous one. The answer to sub-question k is needed to identify the correct document for sub-question k+1. Single-hop shortcuts are systematically removed during dataset construction.

6 types of reasoning graphs covering different compositional structures.

### Size

| Split | Examples |
|---|---:|
| Train | 19,938 |
| Dev | 2,417 |
| Test | 2,459 |
| Total (MuSiQue-Ans) | ~24,814 |

MuSiQue-Full (adds unanswerable contrast questions): ~50,000. Source sub-questions drawn from SQuAD, T-REx, Natural Questions, MLQA, Zero-Shot RE; all backed by Wikipedia.

### Structure

Each example:
- `question`: naturally written multi-hop question (no visible decomposition).
- `answer`: short span (typically 1 to 2 tokens).
- `question_decomposition`: numbered sub-questions Q1...Q_n with individual answers A1...A_n.
- `paragraphs`: supporting Wikipedia paragraphs (one per hop) plus distractors.
- `answerable`: boolean (MuSiQue-Full only).

Answer types: span extraction, yes/no, comparison. Answer length: typically 1 to 2 tokens.

### Example (2-hop, constructed bottom-up)

```
sub-question construction:
  Q1: "Who directed [film X]?"            -> A1: "[Director Y]"
  Q2: "What is [Director Y]'s nationality?" -> A2: "[Country Z]"

composed question: "What is the nationality of the director of [film X]?"
answer: "[Country Z]"
```

The final question is written naturally; no numbered sub-questions are visible to the model. The model must internally decompose and chain the reasoning.

A harder 3-hop example:

```
composed: "Where was the person who founded the publisher of [book X] born?"
hops:
  1. [book X]    -> publisher Y
  2. publisher Y -> founder Z
  3. founder Z   -> birthplace W
answer: W
```

### Difficulty relative to HotpotQA

A single-paragraph baseline achieves ~65 F1 on HotpotQA but only ~32 F1 on MuSiQue (a 30-point drop), because MuSiQue removes the shortcuts that HotpotQA allows. The human-machine gap is 3× larger on MuSiQue. In our eval: GRPO base 0.055 vs paper 0.066; both near floor.

### Project notes

MuSiQue is the hardest tier in the E2H curriculum. The near-floor EM (~0.06) indicates Search-R1 barely solves it, even trained on NQ + HotpotQA. Curriculum fading (ramping MuSiQue weight up while ramping NQ down) is the mechanism E2H uses to avoid the model forgetting easy-task skills while learning to handle 3 to 4 hop chains. This is the central motivation for including E2H in the recipe.

---

## 7. Bamboogle

**Paper**: Press et al., EMNLP 2023 ("Measuring and Narrowing the Compositionality Gap in Language Models", [arxiv 2210.03350](https://arxiv.org/abs/2210.03350))
**Links**: [HuggingFace](https://huggingface.co/datasets/chiayewken/bamboogle) ; introduced as the eval set for Self-Ask.

### Hops

2 hops. Designed adversarially: questions are 2-hop compositions that **a single Google search cannot answer directly**. The dataset is small (125 examples) and used purely as a difficulty stress-test.

### Size

Test-only, **125 examples**. Used full in our eval (no subsampling needed).

### Structure

Each example:
- `id`: string.
- `question`: natural-language 2-hop question (often time-anchored or geography-anchored).
- `golden_answers`: short canonical answer.

### Example (from our `data/bamboogle/test.jsonl`)

```json
{"id": "test_0",
 "question": "Who was president of the United States in the year that Citibank was founded?",
 "golden_answers": ["james madison"]}
```

Reasoning: hop 1 (Citibank founding year: 1812); hop 2 (US president in 1812: James Madison).

### Project notes

Bamboogle is the cheapest meaningful OOD signal in the suite (125 rows, ~6 min on a 4090). However, n=125 makes it noisy: a 3 pp swing is single-sample variance, not signal. Our v1 base EM 0.088 vs paper 0.128 ((−4.0 pp) is within n=125 noise. v1 instruct 0.344 vs paper 0.232 (+11.2 pp) is also noise, not a real lift. Use Bamboogle as a "did the run change anything?" canary, not as a primary metric.

---

## E2H curriculum application to this project

**Paper**: Parashar et al., ICLR 2026, "Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning" ([arxiv 2506.06632](https://arxiv.org/abs/2506.06632))
**Code**: https://github.com/divelab/E2H-Reasoning
**Deep ingest**: [docs/papers/2506.06632_e2h.md](../papers/2506.06632_e2h.md)

### Key method: E2H scheduling

Data partitioned by difficulty; sampling probability shifts from easy to hard over training. Two schedule variants:

**E2H-Cosine (E2H-C)**: cosine interpolation between difficulty levels.

```
S_cosine(t, k) = α_t · (K - k - 1) + (1 - α_t) · k
α_t           = 0.5 · (1 + cos(π · t / T))
```

where `k` = difficulty ordinal (0 = easiest), `t` = current step, `T` = total steps.

**E2H-Gaussian (E2H-G)**: Gaussian mixture with progression speed `β` and concentration `σ`.

```
S_Gaussian(t, k) = exp(- (x_t - μ_k)² / (2 σ²))
x_t              = (t / T)^β · (K - 1)
μ_k              = k - 1
```

Best config from paper: `β = 0.5`, `σ = 0.5`. Higher `β` = faster shift to hard. Lower `σ` = more concentrated sampling.

### Critical finding: fading is essential

Running easy tasks all the way through training hurts. The scheduler must reduce their weight over time. Without fading, easy tasks can dominate and prevent learning on hard examples. This is why stage-based curriculum (fixed 300 steps/stage as in the supervisor meeting proposal) is a reasonable first approximation, but the cosine or Gaussian continuous schedule is the paper's preferred mechanism.

### How difficulty maps to our datasets

Only NQ, HotpotQA, MuSiQue have public training splits, so the E2H ladder uses these three. The other four (TriviaQA, PopQA, 2Wiki, Bamboogle) remain held-out OOD eval.

| Difficulty ordinal | Dataset | Hops | Our v1 base EM |
|---|---|---|---:|
| 0 (easiest) | NQ | 1 | 0.390 |
| 1 (medium) | HotpotQA | 2 | 0.263 |
| 2 (hardest) | MuSiQue | 2 to 4 | 0.055 |

For the retrieval QA setting, hop count is the natural difficulty proxy. No zero-shot error-rate estimation needed (the ordering is clear from EM performance alone), unlike the closed-book math benchmarks where the E2H paper estimates difficulty via base-model pass-rate.

### Important caveat: novelty

The E2H paper uses closed-book math, planning, and coding benchmarks. Applying it to retrieval-augmented multi-hop QA (with live search calls in the GRPO rollout) is a novel extension. The mechanisms should transfer (cold-start bootstrapping, progressive difficulty), but the speedups and convergence rates may differ. This is part of what the thesis validates.

### Hyperparameters (from paper's GRPO config)

```yaml
num_generations: 8         # G=8 rollouts per prompt
beta: 0.001                # KL coefficient (same as our β anneal start)
learning_rate: 1e-6        # matches Search-R1
lr_scheduler_type: cosine
bf16: true
seed: 42
curriculum_schedule: "gaussian"
beta_scheduler: 0.5        # progression speed β
sigma: 0.5                 # concentration
```

### Key results from paper (Qwen 1.5B, hard-split accuracy)

| Method | Blocksworld Hard | Countdown Hard | MATH Hard |
|---|---|---|---|
| GRPO (balanced) | 21.1% | 18.1% | 46.3% |
| E2H-G best | **32.9%** | **41.0%** | **48.7%** |

Largest gains on Blocksworld (planning) and Countdown (arithmetic); both tasks with large easy-to-hard gaps, analogous to NQ→MuSiQue in our setting.

---

## Links to related docs

- [docs/papers/2506.06632_e2h.md](../papers/2506.06632_e2h.md): deep paper ingest of E2H Reasoner.
- [SURVEY_FOCUSED.md](SURVEY_FOCUSED.md): paper cards for datasets and retrieval methods (E2H card at §5.4).
- [PARADIGM_REVIEW.md](PARADIGM_REVIEW.md) §17: full Variant C stack with E2H included.
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md): decision tree for recipe runs.
- [docs/report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md § 6](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md): supervisor-facing recipe proposal.
- [docs/training/CONVERSATION_CONTEXT.md](../training/CONVERSATION_CONTEXT.md): Phase-2 training pipeline status.
- [docs/report/RESULTS_m1.md](../report/RESULTS_m1.md): per-dataset v1 EM vs paper.

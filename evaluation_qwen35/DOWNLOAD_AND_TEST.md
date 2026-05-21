# Download & Test M4.1 Prompt on Mac

Download both Qwen3.5-0.8B models locally and validate the M4.1 prompt rendering + inference.

## Prerequisites

```bash
pip install transformers torch
```

## Run

```bash
cd evaluation_qwen35
python3 download_and_test_models.py
```

## What It Does

**Phase 1: Download Models** (~10 GB total, ~15–30 min over the network)
- Qwen/Qwen3.5-0.8B-Base (~5.3 GB)
- Qwen/Qwen3.5-0.8B (~5.3 GB)
- Models are cached in `~/.cache/huggingface/` by default

**Phase 2: Load Test Questions**
- Loads 5 questions from `data/bamboogle/test.jsonl`

**Phase 3: Test M4.1 Prompt**
- For each model, on each question:
  - Renders prompt via `apply_chat_template` (verifies auto-injected `# Tools` + `<IMPORTANT>`)
  - Generates 200 tokens (CPU inference)
  - Extracts `<tool_call>` blocks and final `<answer>`
  - Logs results

## Expected Output

```
SUMMARY

Qwen3.5-0.8B-Base
  Tests: 5/5 passed
  Tool calls (avg): 1.20/question
  Answers: 5/5 questions answered

Qwen3.5-0.8B
  Tests: 5/5 passed
  Tool calls (avg): 1.40/question
  Answers: 5/5 questions answered

[SUCCESS] M4.1 prompt validation complete
```

## What to Check

✓ **Prompt rendering**: Does `apply_chat_template` inject `# Tools` and `<IMPORTANT>`?
✓ **Tool calls**: Are both models emitting `<tool_call>` blocks with valid query text?
✓ **Answers**: Are both models emitting final `<answer>` blocks?
✓ **Format**: Do the blocks match the canonical nested-XML form?

## Timing on Mac

- Download: 15–30 min (network dependent)
- Inference: ~1–2 min per model per question (CPU is slow; 5 questions × 2 models ≈ 20 min total)
- Total: ~1 hour

## Next

If all tests pass:
1. Commit this test scaffold
2. Proceed to ALICE sbatch smoke tests (100 items × 7 datasets × 2 variants)

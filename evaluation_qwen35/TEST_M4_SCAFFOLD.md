# M4.1 Prompt Validation Scaffold

Local validation of the M4.1 prompt before ALICE smoke tests.

## Goal

Verify that:
1. The Qwen3.5 native nested-XML prompt renders correctly with `apply_chat_template` (auto-injects `# Tools` + `<IMPORTANT>`)
2. Both base and instruct models emit `<tool_call>` blocks
3. Both models can observe `<tool_response>` and reason
4. Both models emit final `<answer>` blocks

## Files

- `test_m4_prompt_scaffold.py` — the scaffold (loads 50 questions, renders prompt, runs local inference on both models)
- This README

## Setup

```bash
# Ensure transformers and torch are installed in your local env
pip install transformers torch

# Verify template + tool schema are importable
python3 -c "from flashrag.search_r1.templates import QWEN35_NATIVE_TEMPLATE, QWEN35_SEARCH_TOOL; print('OK')"
```

## Run

```bash
cd evaluation_qwen35
python3 test_m4_prompt_scaffold.py
```

**Note**: Models are auto-downloaded from HuggingFace on first run (~1.3 GB each). Runs on whatever GPU is available (with `device_map='auto'`). On a 4090, each model takes ~5–10 min to infer on 3 smoke questions.

## Expected Output

- **For each model**: 3 questions, with tool-call extraction and answer detection logged
- **Summary**: tool-calls/question average and answer-rate (%) for base vs instruct
- **Success condition**: both models emit tool_calls (mean > 0.5) and answer_rate > 60%

## Next Steps

1. **Smoke validation passes**: Commit scaffold; proceed to ALICE sbatch (100 items × 7 datasets × 2 variants)
2. **Smoke validation fails**: Debug locally (check prompt rendering, model available, VRAM), then iterate on scaffold
3. **After ALICE smoke**: Populate `docs/report/RESULTS_m4.md` with full sweep results

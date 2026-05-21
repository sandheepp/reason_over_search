#!/usr/bin/env python3
"""
M4.1 Prompt Test for Mac — HF Transformers (CPU OK, no SGLang needed)
Tests prompt rendering via apply_chat_template + optional light inference.
Usage: python3 test_m4_prompt_mac.py [--inference] [--sample-size N]
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional
import random

try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: transformers not available. Install: pip install transformers")
    sys.exit(1)

# Import the M4.1 template and tool schema
sys.path.insert(0, str(Path(__file__).parent))
from flashrag.search_r1.templates import QWEN35_NATIVE_TEMPLATE, QWEN35_SEARCH_TOOL


def load_questions(data_path: str = "../data/bamboogle/test.jsonl", sample_size: int = 5) -> List[Dict]:
    """Load and sample questions from test corpus."""
    questions = []
    with open(data_path, "r") as f:
        for line in f:
            questions.append(json.loads(line))

    if len(questions) < sample_size:
        print(f"WARNING: Only {len(questions)} available; using all.")
        sample_size = len(questions)

    sampled = random.sample(questions, sample_size)
    print(f"[LOAD] {len(sampled)} questions from {Path(data_path).name}")
    return sampled


def test_prompt_render(question: str, model_id: str = "Qwen/Qwen3.5-0.8B") -> Dict:
    """
    Test prompt rendering via apply_chat_template (no inference).
    This validates that the prompt is well-formed and auto-injection works.
    """
    print(f"\n{'─' * 70}")
    print(f"Question: {question[:70]}")
    print(f"Model: {model_id.split('/')[-1]}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"[LOAD] Tokenizer OK")
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}")
        return {"status": "error", "reason": str(e)}

    # Build messages
    messages = [
        {
            "role": "system",
            "content": QWEN35_NATIVE_TEMPLATE,
        },
        {
            "role": "user",
            "content": f"Question: {question}",
        },
    ]

    # Apply chat template with search tool
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tools=[QWEN35_SEARCH_TOOL],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"[RENDER] OK ({len(formatted)} chars)")
    except Exception as e:
        print(f"[ERROR] apply_chat_template failed: {e}")
        return {"status": "error", "reason": str(e)}

    # Validate structure
    has_system = QWEN35_NATIVE_TEMPLATE[:20] in formatted  # rough check
    has_question = f"Question: {question[:20]}" in formatted
    has_tools_block = "# Tools" in formatted
    has_important = "<IMPORTANT>" in formatted
    has_think_prefix = "<think>" in formatted

    print(f"[VALIDATE] system message: {has_system}")
    print(f"[VALIDATE] question prefix: {has_question}")
    print(f"[VALIDATE] auto-injected # Tools block: {has_tools_block}")
    print(f"[VALIDATE] auto-injected <IMPORTANT> reminder: {has_important}")
    print(f"[VALIDATE] <think> prefix for reasoning: {has_think_prefix}")

    # Show a snippet
    print(f"\n[SNIPPET] First 500 chars of rendered prompt:")
    print("─" * 70)
    print(formatted[:500])
    print("─" * 70)

    return {
        "status": "ok",
        "length": len(formatted),
        "has_system": has_system,
        "has_question": has_question,
        "has_tools_block": has_tools_block,
        "has_important": has_important,
        "has_think_prefix": has_think_prefix,
    }


def test_inference_light(question: str, model_id: str = "Qwen/Qwen3.5-0.8B") -> Dict:
    """
    Light inference test (CPU, max_tokens=100 for speed on Mac).
    Validates parsing of tool-call and answer blocks.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("[SKIP] torch not available; skipping inference test")
        return {"status": "skipped"}

    print(f"\n[INFER] Running light inference (max_tokens=100)...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Load on CPU for Mac compatibility
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        print(f"[LOAD] Model loaded on CPU")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return {"status": "error", "reason": str(e)}

    messages = [
        {"role": "system", "content": QWEN35_NATIVE_TEMPLATE},
        {"role": "user", "content": f"Question: {question}"},
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tools=[QWEN35_SEARCH_TOOL],
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        inputs = tokenizer(formatted, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"[GEN] Generated {outputs.shape[1]} tokens")
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return {"status": "error", "reason": str(e)}

    # Extract response portion
    response_start = response.rfind("<|im_start|>assistant\n")
    if response_start != -1:
        response = response[response_start + len("<|im_start|>assistant\n"):]

    # Validate structure
    tool_calls = re.findall(
        r"<tool_call>.*?<parameter=query>\s*(.*?)\s*</parameter>.*?</tool_call>",
        response,
        re.DOTALL,
    )
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)

    print(f"[VALIDATE] Tool calls emitted: {len(tool_calls)}")
    if tool_calls:
        for i, call in enumerate(tool_calls, 1):
            print(f"           [{i}] {call[:40]}...")
    print(f"[VALIDATE] Final answer: {answer.group(1)[:40] if answer else '(none)'}...")

    print(f"\n[SNIPPET] Assistant response (first 300 chars):")
    print("─" * 70)
    print(response[:300])
    print("─" * 70)

    del model
    torch.cuda.empty_cache()

    return {
        "status": "ok",
        "tool_calls": len(tool_calls),
        "has_answer": bool(answer),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="M4.1 Prompt test for Mac (HF Transformers, CPU OK)"
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Also run light inference (slow on CPU; CPU-only, 100 tokens)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of questions to test (default 3 for quick smoke on Mac)",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default="Qwen/Qwen3.5-0.8B-Base",
        help="Base model ID",
    )
    parser.add_argument(
        "--model-instruct",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="Instruct model ID",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("M4.1 Prompt Test — Mac (HF Transformers)")
    print("=" * 70)
    print(f"[CONFIG] render_test=True, inference_test={args.inference}")
    print(f"[CONFIG] sample_size={args.sample_size} (default 3 for Mac speed)")
    print(f"[CONFIG] base={args.model_base.split('/')[-1]}")
    print(f"[CONFIG] instruct={args.model_instruct.split('/')[-1]}")

    # Load questions
    questions = load_questions(sample_size=args.sample_size)

    models = [args.model_base, args.model_instruct]
    render_results = {}
    infer_results = {}

    # Test prompt rendering (fast, CPU OK)
    print(f"\n{'=' * 70}")
    print("PHASE 1: Prompt Rendering (apply_chat_template)")
    print(f"{'=' * 70}")

    for model_id in models:
        model_name = model_id.split("/")[-1]
        render_results[model_name] = []

        for q in questions:
            result = test_prompt_render(q["question"], model_id)
            render_results[model_name].append(result)

    # Test inference (optional, slow on Mac CPU)
    if args.inference:
        print(f"\n{'=' * 70}")
        print("PHASE 2: Light Inference (100 tokens, CPU mode)")
        print("(This will be SLOW on Mac; patience required)")
        print(f"{'=' * 70}")

        for model_id in models:
            model_name = model_id.split("/")[-1]
            infer_results[model_name] = []

            for q in questions:
                result = test_inference_light(q["question"], model_id)
                infer_results[model_name].append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for model_name, results in render_results.items():
        ok_count = sum(1 for r in results if r.get("status") == "ok")
        print(
            f"\n{model_name:30} Render tests: {ok_count}/{len(results)} OK"
        )
        if results and results[0].get("status") == "ok":
            r = results[0]
            print(f"  └─ tools_block: {r.get('has_tools_block')}, "
                  f"IMPORTANT: {r.get('has_important')}, "
                  f"think: {r.get('has_think_prefix')}")

    if infer_results:
        for model_name, results in infer_results.items():
            ok_count = sum(1 for r in results if r.get("status") == "ok")
            print(
                f"\n{model_name:30} Infer tests: {ok_count}/{len(results)} OK"
            )
            if results and results[0].get("status") == "ok":
                r = results[0]
                print(f"  └─ tool_calls: {r.get('tool_calls', 0)}, "
                      f"answer: {r.get('has_answer')}")

    print(f"\n[SUCCESS] M4.1 prompt test complete")
    print(f"[NEXT] If all render tests pass, proceed to ALICE sbatch")


if __name__ == "__main__":
    main()

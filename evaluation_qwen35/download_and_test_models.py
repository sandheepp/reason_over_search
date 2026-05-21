#!/usr/bin/env python3
"""
Download Qwen3.5-0.8B models and test M4.1 prompt locally on Mac.
Usage: python3 download_and_test_models.py
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional
import random

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("ERROR: transformers or torch not installed")
    print("Install: pip install transformers torch")
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


def download_model(model_id: str) -> bool:
    """Download model and tokenizer; cache locally."""
    print(f"\n{'─' * 70}")
    print(f"Downloading {model_id}...")
    print(f"{'─' * 70}")

    try:
        print(f"[DL] Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"[DL] ✓ Tokenizer OK")

        print(f"[DL] Model (~5.3 GB for 0.8B)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        print(f"[DL] ✓ Model OK ({model.config.model_type})")

        del model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def test_model(model_id: str, questions: List[Dict]) -> Dict:
    """Test M4.1 prompt on a model."""
    model_name = model_id.split("/")[-1]
    print(f"\n{'=' * 70}")
    print(f"Testing {model_name}")
    print(f"{'=' * 70}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    except Exception as e:
        print(f"[ERROR] Failed to load {model_id}: {e}")
        return {"status": "error", "reason": str(e)}

    results = {
        "model_id": model_id,
        "tests": [],
        "tool_calls_total": 0,
        "answers_total": 0,
    }

    for i, q in enumerate(questions, 1):
        question = q["question"]
        print(f"\n[TEST {i}/{len(questions)}] {question[:60]}...")

        # Build prompt
        messages = [
            {"role": "system", "content": QWEN35_NATIVE_TEMPLATE},
            {"role": "user", "content": f"Question: {question}"},
        ]

        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tools=[QWEN35_SEARCH_TOOL],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"  [ERROR] apply_chat_template failed: {e}")
            results["tests"].append({"status": "error", "reason": str(e)})
            continue

        # Generate
        try:
            inputs = tokenizer(formatted, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        except Exception as e:
            print(f"  [ERROR] Generation failed: {e}")
            results["tests"].append({"status": "error", "reason": str(e)})
            continue

        # Extract assistant response
        response_start = response.rfind("<|im_start|>assistant\n")
        if response_start != -1:
            response = response[response_start + len("<|im_start|>assistant\n"):]

        # Parse
        tool_calls = re.findall(
            r"<tool_call>.*?<parameter=query>\s*(.*?)\s*</parameter>.*?</tool_call>",
            response,
            re.DOTALL,
        )
        answer = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)

        print(f"  ✓ Tool calls: {len(tool_calls)}", end="")
        if tool_calls:
            print(f" [{tool_calls[0][:30]}...]")
        else:
            print()

        if answer:
            ans_text = answer.group(1).strip()[:40]
            print(f"  ✓ Answer: {ans_text}...")
            results["answers_total"] += 1
        else:
            print(f"  ✗ Answer: (none)")

        results["tool_calls_total"] += len(tool_calls)
        results["tests"].append({
            "status": "ok",
            "question": question[:50],
            "tool_calls": len(tool_calls),
            "has_answer": bool(answer),
        })

    del model
    torch.cuda.empty_cache()

    return results


def main():
    print("=" * 70)
    print("M4.1 Prompt Test — Download & Validate on Mac")
    print("=" * 70)

    models = [
        "Qwen/Qwen3.5-0.8B-Base",
        "Qwen/Qwen3.5-0.8B",
    ]

    # Download phase
    print("\nPHASE 1: Download Models")
    print("=" * 70)

    downloaded = {}
    for model_id in models:
        success = download_model(model_id)
        downloaded[model_id] = success

    if not all(downloaded.values()):
        print("\n[ABORT] Some downloads failed. Fix errors above and retry.")
        sys.exit(1)

    # Load questions
    print("\n\nPHASE 2: Load Test Questions")
    print("=" * 70)
    questions = load_questions(sample_size=5)

    # Test phase
    print("\n\nPHASE 3: Test M4.1 Prompt on Both Models")
    print("=" * 70)

    test_results = {}
    for model_id in models:
        if downloaded[model_id]:
            test_results[model_id] = test_model(model_id, questions)

    # Summary
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}\n")

    for model_id, results in test_results.items():
        model_name = model_id.split("/")[-1]
        n_tests = len(results["tests"])
        ok_tests = sum(1 for t in results["tests"] if t.get("status") == "ok")

        print(f"{model_name:30}")
        print(f"  Tests: {ok_tests}/{n_tests} passed")
        print(f"  Tool calls (avg): {results['tool_calls_total'] / n_tests:.2f}/question")
        print(f"  Answers: {results['answers_total']}/{n_tests} questions answered")

    print(f"\n[SUCCESS] M4.1 prompt validation complete")
    print(f"[NEXT] If tests pass, proceed to ALICE sbatch smoke tests")


if __name__ == "__main__":
    main()

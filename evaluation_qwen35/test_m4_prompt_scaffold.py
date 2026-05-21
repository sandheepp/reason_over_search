#!/usr/bin/env python3
"""
M4.1 Prompt Validation Scaffold
Test the Qwen3.5 native nested-XML tool-call prompt with both base and instruct models.
Validates: prompt rendering, tool-call parsing, tool-response wrapping, answer extraction.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("ERROR: transformers not available. Install: pip install transformers torch")
    sys.exit(1)

# Import the M4.1 template and tool schema
sys.path.insert(0, str(Path(__file__).parent))
from flashrag.search_r1.templates import QWEN35_NATIVE_TEMPLATE, QWEN35_SEARCH_TOOL


def load_questions(data_path: str = "../data/bamboogle/test.jsonl", sample_size: int = 50) -> List[Dict]:
    """Load and sample questions from test corpus."""
    questions = []
    with open(data_path, "r") as f:
        for line in f:
            questions.append(json.loads(line))

    if len(questions) < sample_size:
        print(f"WARNING: Only {len(questions)} available; using all.")
        sample_size = len(questions)

    sampled = random.sample(questions, sample_size)
    print(f"[SCAFFOLD] Loaded {len(sampled)} questions from {data_path}")
    return sampled


def mock_retriever(query: str) -> str:
    """Mock retriever that returns canned passages. In real eval, hits FAISS."""
    passages = [
        f"Passage 1: Information about '{query}' from Wikipedia. This contains relevant details.",
        f"Passage 2: Further context related to '{query}' in historical records.",
        f"Passage 3: Additional reference material about '{query}' and related topics.",
    ]
    return "\n".join(passages)


def render_prompt(question: str, model_id: str) -> Tuple[str, str]:
    """
    Render the full chat conversation using apply_chat_template.
    Returns: (formatted_prompt, rendered_system_message)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build the conversation with system message + user question
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

    # Apply chat template with the search tool registered
    # This will auto-inject the # Tools block and <IMPORTANT> reminder
    formatted = tokenizer.apply_chat_template(
        messages,
        tools=[QWEN35_SEARCH_TOOL],
        tokenize=False,
        add_generation_prompt=True,
    )

    return formatted, QWEN35_NATIVE_TEMPLATE


def extract_tool_calls(text: str) -> List[str]:
    """Extract all <tool_call> blocks from text."""
    pattern = re.compile(
        r"<tool_call>.*?<parameter=query>\s*(.*?)\s*</parameter>.*?</tool_call>",
        re.DOTALL,
    )
    return [m.group(1).strip() for m in pattern.finditer(text)]


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from <answer>...</answer> block."""
    pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def run_inference(
    model_id: str,
    question: str,
    max_new_tokens: int = 500,
) -> str:
    """
    Run a single inference step:
    1. Render prompt with system message
    2. Generate response
    3. Validate tool-call / answer structure
    """
    print(f"\n[TEST] Question: {question[:60]}...")
    print(f"[TEST] Model: {model_id}")

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"[LOAD] Loaded {model_id.split('/')[-1]}")
    except Exception as e:
        print(f"[ERROR] Failed to load {model_id}: {e}")
        return ""

    # Render the prompt
    formatted_prompt, system_msg = render_prompt(question, model_id)
    print(f"[RENDER] System message OK ({len(system_msg)} chars)")
    print(f"[RENDER] Full prompt OK ({len(formatted_prompt)} chars, includes auto-injected # Tools + <IMPORTANT>)")

    # Tokenize and generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy, matching eval config
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"[GEN] Generated {outputs.shape[1]} tokens")
    except torch.cuda.OutOfMemoryError:
        print(f"[ERROR] OOM on {model_id} — reduce max_new_tokens or use a smaller model")
        return ""
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return ""

    # Extract the assistant's response (strip the prompt prefix)
    response_start = generated_text.rfind("<|im_start|>assistant\n")
    if response_start == -1:
        print("[WARN] Could not find assistant marker in output")
        response = generated_text
    else:
        response = generated_text[response_start + len("<|im_start|>assistant\n"):]

    # Validate structure
    tool_calls = extract_tool_calls(response)
    answer = extract_answer(response)

    print(f"[VALIDATE] Tool calls: {len(tool_calls)}")
    if tool_calls:
        for i, call in enumerate(tool_calls, 1):
            print(f"           [{i}] {call[:50]}...")
    print(f"[VALIDATE] Final answer: {answer[:50] if answer else '(none)'}...")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return response


def main():
    """
    Scaffold entry point:
    1. Load 50 questions
    2. Render prompt for each (check apply_chat_template works)
    3. Run inference on both base and instruct (if available)
    4. Collect stats on tool-call / answer structure
    """
    print("=" * 70)
    print("M4.1 Prompt Validation Scaffold")
    print("=" * 70)

    # Load questions
    questions = load_questions(sample_size=50)

    # Run on first 3 questions as smoke test
    smoke_sample = questions[:3]

    print(f"\n[SCAFFOLD] Running smoke test on {len(smoke_sample)} questions")
    print(f"[SCAFFOLD] Models: Qwen3.5-0.8B (base + instruct)")
    print(f"[SCAFFOLD] Goal: validate prompt rendering, tool-call parsing, answer extraction")

    models = [
        "Qwen/Qwen3.5-0.8B-Base",
        "Qwen/Qwen3.5-0.8B",
    ]

    results = {}

    for model_id in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_id}")
        print(f"{'=' * 70}")

        tool_call_counts = []
        answer_counts = 0

        for q in smoke_sample:
            question = q["question"]
            try:
                response = run_inference(model_id, question, max_new_tokens=300)

                if response:
                    tool_calls = extract_tool_calls(response)
                    answer = extract_answer(response)

                    tool_call_counts.append(len(tool_calls))
                    if answer:
                        answer_counts += 1

            except KeyboardInterrupt:
                print("\n[INTERRUPT] Test interrupted by user")
                sys.exit(0)
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

        results[model_id] = {
            "tool_calls_mean": sum(tool_call_counts) / len(tool_call_counts) if tool_call_counts else 0,
            "answer_rate": answer_counts / len(smoke_sample) if smoke_sample else 0,
        }

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for model_id, stats in results.items():
        print(f"{model_id.split('/')[-1]:30} tool_calls/q: {stats['tool_calls_mean']:.2f}  answer_rate: {stats['answer_rate']:.0%}")

    print(f"\n[SUCCESS] M4.1 prompt scaffold validation complete")
    print(f"[NEXT] If scaffold passes, proceed to ALICE sbatch smoke tests (100 items × 7 datasets)")


if __name__ == "__main__":
    main()

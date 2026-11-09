#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.24",
#   "pyarrow>=15",
# ]
# ///
"""Download the Search-R1 RL training corpus and reshape it for NeMo-RL.

Source : https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train
Output : data/training/nq_hotpotqa_train/{train,test}.parquet (repo-root relative)

Two transforms in one pass:

1. **Strip the prebaked Search-R1 template.** Upstream `make_prefix` bakes the
   paper's `<think>`/`<search>`/`<information>`/`<answer>` instruction string
   into `prompt[0].content`. We keep the dataset template-agnostic so the
   chat template (Qwen3.5 native `<tool_call>` default vs. the paper's
   `<search>` ablation arm) is applied at rollout time via run config — see
   docs/training/CHAT_TEMPLATE.md.

2. **Rename `prompt` → `messages`.** NeMo-RL's ResponseDataset routes on the
   `messages` column (nemo_rl/data/datasets/response_datasets/response_dataset.py:64):
   present → preserve all other columns + add `task_name`; absent → drop
   everything except input/output. We want the preserve branch so
   `golden_answers`, `data_source`, etc. survive into the processor.

Each output row's `messages` is a length-1 list `[{"role": "user", "content": question}]`.
We don't add an `assistant` turn — Search-R1 has no reference rollout, only a
gold-answer list, which we read from the `golden_answers` column directly in
the M2-step-4 custom processor (rather than encoding it into messages[1]).

We do NOT pre-bake `task_name`. NeMo-RL's ResponseDataset adds it via
`add_column`, which would conflict if the column already existed. The
M2-step-4 dataset adapter sets the proper task name.

Preserved verbatim: `id`, `question`, `golden_answers`, `data_source`,
`reward_model`, `extra_info`, `metadata`, `ability`.

We read the upstream parquets directly with pyarrow rather than via
`datasets.load_dataset`, because the latter tries to unify the per-row
`extra_info`/`metadata` schema across the NQ + HotpotQA mixture and fails.

Idempotent. Re-runs skip if the output parquets exist; pass `--force` to override.

Run:
  training/scripts/prepare_dataset.py [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HF_REPO = "PeterJinGo/nq_hotpotqa_train"
SPLITS = ("train", "test")
EXPECTED_ROWS = {"train": 169_615, "test": 51_713}

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "training" / "nq_hotpotqa_train"


def reshape_for_nemo_rl(table: pa.Table) -> pa.Table:
    """Drop `prompt`, add `messages: [{"role": "user", "content": question}]`."""
    questions = table.column("question").to_pylist()
    rows = [[{"role": "user", "content": q}] for q in questions]
    messages_type = table.schema.field("prompt").type  # list<struct<role, content>>
    return table.drop_columns(["prompt"]).append_column(
        "messages", pa.array(rows, type=messages_type)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--force", action="store_true", help="re-download even if outputs exist")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = {s: OUT_DIR / f"{s}.parquet" for s in SPLITS}

    if not args.force and all(p.exists() for p in out_paths.values()):
        print(f"[skip] outputs already present at {OUT_DIR} (use --force to redo)")
        for s, p in out_paths.items():
            print(f"  {s}: {p} ({p.stat().st_size / 1e6:.1f} MB)")
        return 0

    sample_before: str | None = None
    sample_after: str | None = None
    sample_question: str | None = None

    for split in SPLITS:
        print(f"[fetch] {HF_REPO}:{split}.parquet")
        src = hf_hub_download(HF_REPO, f"{split}.parquet", repo_type="dataset")
        table = pq.read_table(src)
        n = table.num_rows
        if n != EXPECTED_ROWS[split]:
            print(f"[warn] {split}: got {n} rows, expected {EXPECTED_ROWS[split]}")

        if split == "train":
            sample_before = table.column("prompt")[0].as_py()[0]["content"]

        reshaped = reshape_for_nemo_rl(table)

        if split == "train":
            sample_after = reshaped.column("messages")[0].as_py()[0]["content"]
            sample_question = reshaped.column("question")[0].as_py()
            print(f"[schema] {split} columns: {reshaped.column_names}")

        pq.write_table(reshaped, out_paths[split])
        print(f"[write] {split}: {out_paths[split]} ({n} rows, {out_paths[split].stat().st_size / 1e6:.1f} MB)")

    print("\n[verify] row 0 messages[0].content")
    print(f"  upstream prompt[0].content: {sample_before[:120]}{'…' if len(sample_before) > 120 else ''}")
    print(f"  ours messages[0].content  : {sample_after!r}")
    print(f"  ours question             : {sample_question!r}")
    if sample_after != sample_question:
        print("[error] post-convert messages content does not match question", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

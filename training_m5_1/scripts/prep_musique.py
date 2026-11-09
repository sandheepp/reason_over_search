#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.24",
#   "pyarrow>=15",
# ]
# ///
"""Pull the MuSiQue training split and reshape it for NeMo-RL.

Source : RUC-NLPIR/FlashRAG_datasets (musique/train.jsonl) on HuggingFace.
         Same publisher / schema as the dev.jsonl that the M4 eval pipeline
         reads from `data/musique/dev.jsonl`, so train and eval rows are
         schema-identical.
Output : data/training/musique/train.parquet (repo-root relative).

Schema:
  in  (jsonl rows): {id, question, golden_answers: list[str], metadata: dict}
  out (parquet)   : {id, question, golden_answers, data_source, messages}
                    where messages = [{"role": "user", "content": question}]
                    and   data_source = "musique" (lets the processor log
                                                   per-dataset stats once
                                                   M5.2 mixes in HotpotQA/2Wiki)

Why parquet (and not jsonl, which is what eval reads): NeMo-RL's
`SearchR1Dataset` reads via `datasets.Dataset.from_parquet` (parquet preserves
list+dict columns without re-parsing); we keep the M2 contract and just
swap the file. The eval pipeline keeps its own jsonl files at
`data/musique/{dev,test}.jsonl` — independent paths, no overlap.

In-training validation: NOT carved here. `SearchR1Dataset(split_validation_size=...)`
handles the train/val carve at load time so the seed is config-controlled
(grpo.seed). Keeping all rows in train.parquet preserves that knob.

Idempotent: re-runs skip if the output exists; pass `--force` to override.
Manual run (offline-safe — needs HF connectivity):
  uv run training_m5_1/scripts/prep_musique.py [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HF_REPO = "RUC-NLPIR/FlashRAG_datasets"
HF_FILE = "musique/train.jsonl"
DATA_SOURCE_TAG = "musique"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "training" / "musique"
OUT_PATH = OUT_DIR / "train.parquet"

# Expected upstream size; loose check (a future FlashRAG release could trim
# stragglers). Hard-fails if the row count is wildly off, soft-warns otherwise.
EXPECTED_ROWS_LOOSE = (15_000, 22_000)  # MuSiQue train is ~19_938


def reshape(rows: list[dict]) -> pa.Table:
    """Build the NeMo-RL parquet from the raw jsonl rows."""
    ids: list[str] = []
    questions: list[str] = []
    golden: list[list[str]] = []
    messages: list[list[dict]] = []
    data_source: list[str] = []
    for r in rows:
        q = r["question"]
        ids.append(r["id"])
        questions.append(q)
        golden.append(list(r["golden_answers"]))
        messages.append([{"role": "user", "content": q}])
        data_source.append(DATA_SOURCE_TAG)
    return pa.table(
        {
            "id": ids,
            "question": questions,
            "golden_answers": golden,
            "data_source": data_source,
            "messages": messages,
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--force", action="store_true", help="re-download even if outputs exist")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists() and not args.force:
        print(f"[skip] {OUT_PATH} already present ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
        print(f"       use --force to redo")
        return 0

    print(f"[fetch] {HF_REPO}:{HF_FILE}")
    local = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    rows: list[dict] = []
    with open(local, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n = len(rows)
    lo, hi = EXPECTED_ROWS_LOOSE
    if n < lo or n > hi:
        print(f"[error] got {n} rows; expected roughly {lo}..{hi}", file=sys.stderr)
        return 1
    print(f"[load]  {n} rows")

    table = reshape(rows)
    pq.write_table(table, OUT_PATH)
    print(f"[write] {OUT_PATH} ({n} rows, {OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"[schema] {table.column_names}")

    # Spot-check row 0.
    print()
    print("[verify] row 0:")
    print(f"  id            = {rows[0]['id']!r}")
    print(f"  question      = {rows[0]['question']!r}")
    print(f"  golden_answers= {rows[0]['golden_answers']!r}")
    print(f"  messages[0]   = {table.column('messages')[0].as_py()[0]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

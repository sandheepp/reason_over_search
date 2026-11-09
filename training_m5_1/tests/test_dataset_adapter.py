"""Dataset adapter — `SearchR1Dataset(data_path, arm)` loads parquet correctly.

Skips when `datasets` (HF) or `nemo_rl` aren't installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("datasets")
pytest.importorskip("nemo_rl")

from training_m5_1.src.datasets.search_r1 import SearchR1Dataset, VALID_ARMS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_PARQUET = REPO_ROOT / "data" / "training" / "nq_hotpotqa_train" / "test.parquet"


@pytest.fixture(scope="module")
def parquet_path():
    if not TEST_PARQUET.exists():
        pytest.skip(f"{TEST_PARQUET} not present (run training/scripts/prepare_dataset.py)")
    return str(TEST_PARQUET)


@pytest.mark.parametrize("arm", VALID_ARMS)
def test_loads_parquet_and_sets_task_name(parquet_path, arm):
    ds = SearchR1Dataset(data_path=parquet_path, arm=arm)
    assert ds.task_name == f"search_r1_{arm}"
    assert ds.arm == arm
    assert len(ds.dataset) > 0


def test_preserves_required_columns(parquet_path):
    ds = SearchR1Dataset(data_path=parquet_path, arm="qwen_native")
    cols = set(ds.dataset.column_names)
    # The processor needs at least these:
    assert "messages" in cols
    assert "golden_answers" in cols
    assert "data_source" in cols


def test_messages_shape_is_length_one_user_turn(parquet_path):
    ds = SearchR1Dataset(data_path=parquet_path, arm="qwen_native")
    row = ds.dataset[0]
    assert isinstance(row["messages"], list)
    assert len(row["messages"]) == 1
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][0]["content"] == row["question"] if "question" in row else True


def test_unknown_arm_raises(parquet_path):
    with pytest.raises(ValueError):
        SearchR1Dataset(data_path=parquet_path, arm="nonexistent")


def test_validation_split(parquet_path):
    ds = SearchR1Dataset(data_path=parquet_path, arm="qwen_native", split_validation_size=0.1, seed=42)
    assert ds.val_dataset is not None
    assert len(ds.val_dataset) > 0
    # Train shrinks by ~10%.
    assert len(ds.dataset) > 0

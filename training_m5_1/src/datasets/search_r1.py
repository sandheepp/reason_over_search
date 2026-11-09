"""Search-R1 RawDataset adapter.

Loads our pre-reshaped parquet (data/training/nq_hotpotqa_train/{train,test}.parquet)
into NeMo-RL's `RawDataset` interface. The parquet already has the canonical
`messages: [{"role": "user", "content": question}]` shape that
`ResponseDataset`'s preserve branch routes on (response_dataset.py:64), so we
inherit `RawDataset` directly rather than going through `ResponseDataset` —
this avoids `add_column` collisions and gives us control over `task_name`.

`task_name` is set to `f"search_r1_{arm}"` so the same parquet feeds either
chat-template arm via different task-name routing in PROCESSOR_REGISTRY and
task_to_env. The arm is supplied via the data config:

    data:
      train:
        - dataset_name: search_r1
          data_path: data/training/nq_hotpotqa_train/train.parquet
          arm: qwen_native    # or "paper"
          env_name: search_r1
          processor: search_r1_processor
"""
from __future__ import annotations

from typing import Optional

from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

VALID_ARMS = ("qwen_native", "paper")


class SearchR1Dataset(RawDataset):
    """Wraps the reshaped Search-R1 parquet.

    Args:
        data_path: Path to the parquet file (e.g.
            `data/training/nq_hotpotqa_train/train.parquet`).
        arm: Chat-template arm — `"qwen_native"` (default) or `"paper"`.
            Drives task_name routing; the processor and env both dispatch on it.
        split_validation_size: If >0, carve a validation split from the loaded
            dataset (passed straight through to `RawDataset.split_train_validation`).
        seed: Seed for the train/validation split.
        **kwargs: Other fields from the data config (`env_name`, `processor`,
            `prompt_file`, `system_prompt_file`, ...) — accepted and ignored
            here; consumed by `set_task_spec` / `set_processor` later.
    """

    def __init__(
        self,
        data_path: str,
        arm: str = "qwen_native",
        split_validation_size: float = 0.0,
        seed: int = 42,
        **kwargs,
    ) -> None:
        if arm not in VALID_ARMS:
            raise ValueError(f"arm must be one of {VALID_ARMS}, got {arm!r}")
        self.arm = arm
        self.task_name = f"search_r1_{arm}"

        # `Dataset.from_parquet` keeps every column verbatim — `messages`,
        # `golden_answers`, `data_source`, `id`, etc. all flow through to
        # `AllTaskProcessedDataset.__getitem__`'s `entry` (see processed_dataset.py:101).
        self.dataset: Dataset = Dataset.from_parquet(data_path)
        # processed_dataset.py:107/118 reads entry["task_name"] when
        # task_data_processors is a dict. Stamp it onto every row.
        if "task_name" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                "task_name", [self.task_name] * len(self.dataset)
            )
        self.val_dataset: Optional[Dataset] = None
        self.split_train_validation(split_validation_size, seed)

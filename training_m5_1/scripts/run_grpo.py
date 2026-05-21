#!/usr/bin/env python
"""Search-R1 overlay launcher for NeMo-RL GRPO.

Thin wrapper around `nemo_rl/examples/run_grpo.py` that registers our
Search-R1 dataset / processor / env into NeMo-RL's pluggable registries
before training starts. Doing it here (vs. in the upstream script) keeps the
vendored NeMo-RL source untouched.

Usage (delegates argv to the upstream parser, including Hydra overrides):

    training_m5_1/scripts/run_grpo.py \\
        --config=training_m5_1/configs/m5_1_research_paper.yaml \\
        policy.model_name=Qwen/Qwen3.5-0.8B \\
        grpo.seed=42

Run from the repo root so relative `data_path: data/musique/...` resolves
correctly. The bash wrapper (`scripts/run.sh`) handles the cd + venv
activation.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _setup_paths() -> None:
    """Make `training_m5_1.src` and `examples.run_grpo` importable."""
    paths = (REPO_ROOT, REPO_ROOT / "training_m5_1" / "nemo_rl")
    for p in paths:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def main() -> None:
    _setup_paths()

    # Side effect: registers DATASET_REGISTRY["search_r1"], the search_r1
    # processor, and the search_r1 env. Idempotent — re-import is a no-op.
    import training_m5_1.src.registry  # noqa: F401

    # Hand off to the vendored launcher's main(); it parses argv (Hydra config
    # path + overrides) and runs the full GRPO loop.
    from examples.run_grpo import main as run_grpo_main

    run_grpo_main()


if __name__ == "__main__":
    main()

"""Test config — make `training.src` and `evaluation_search_r1.flashrag` importable.

These tests run in a venv that doesn't have NeMo-RL or torch — so the env
test stubs out `ray.remote`, the `nemo_rl.*` imports, and `torch` before any
overlay module is imported. See test_env_step.py for the stubbing.

Reward-parity and parser tests don't need any of those stubs and import
training.src directly.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Make training/ importable as `training.src` and the eval pipeline as
# `evaluation_search_r1.flashrag.search_r1.reward` so reward-parity tests
# can compare ours to upstream.
for p in (REPO_ROOT, REPO_ROOT / "evaluation_search_r1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

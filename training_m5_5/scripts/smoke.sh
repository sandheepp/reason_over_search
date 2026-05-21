#!/usr/bin/env bash
# Convenience alias for `run.sh --mode smoke`. Same args as run.sh otherwise.
#
# Usage:
#   bash training_m5_5/scripts/smoke.sh                # seed 42
#   bash training_m5_5/scripts/smoke.sh --seed 7
#   bash training_m5_5/scripts/smoke.sh -- policy.train_micro_batch_size=8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run.sh" --mode smoke "$@"

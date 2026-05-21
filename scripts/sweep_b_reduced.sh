#!/usr/bin/env bash
# PLAN B — Reduced 1-day sweep:
#   * Single seed.
#   * Bamboogle (125) and MuSiQue (2,417) at full size.
#   * NQ / TriviaQA / PopQA / HotpotQA / 2Wiki subsampled to 1,000 rows each.
#   * Both model variants.
# Total: ~7,500 examples × 2 variants ≈ 15,000 example evals → ~12–18 h on this box.
#
# Usage:
#   nohup scripts/sweep_b_reduced.sh > /tmp/sweep_b.log 2>&1 &
#   tail -f /tmp/sweep_b.log

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

N=1000
SUB_SEED=42        # Subsample seed for reproducible random sample
SEEDS="1"          # One inference seed
DATASETS="bamboogle nq triviaqa popqa musique 2wikimultihopqa hotpotqa"

echo "=== PLAN B: 1 seed × 7 datasets × 2 variants, N=$N for large datasets ==="
date

echo "[plan-b] preparing data_subsample/"
"$REPO_ROOT/scripts/subsample.sh" "$N" "$SUB_SEED"

for variant in base instruct; do
  echo
  echo "=== switching SGLang to $variant ==="
  "$REPO_ROOT/scripts/manage_sglang.sh" switch "$variant"
  echo "=== running on $variant ==="
  "$REPO_ROOT/scripts/run_variant_sweep.sh" "$variant" "$SEEDS" "$DATASETS" \
      "$REPO_ROOT/data_subsample"
done

echo
echo "=== aggregating ==="
"$REPO_ROOT/scripts/aggregate.py" --output "$REPO_ROOT/docs/report/RESULTS_m1.md"
echo "done — see docs/report/RESULTS_m1.md"
date

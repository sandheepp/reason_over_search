#!/usr/bin/env bash
# Validate that the retriever on $PORT is healthy AND that the async fix
# in retriever_serving.py actually parallelizes concurrent requests.
#
# Logic: send N requests sequentially, then N requests in parallel; report
# the wall-clock for each. With the async fix (def search + FastAPI thread
# pool), parallel ≈ 1× single-request latency (~3-8× speedup over sequential).
# Without it (async def search + sync FAISS, event-loop blocked), parallel
# ≈ sequential; that's the placebo state when the async fix is not active.
#
# Usage:
#   local_retriever/smoke_concurrent.sh [port=3005] [n=8]
#
# Exits non-zero if /health fails or parallel speedup < 2× (clear fix-broken signal).

set -euo pipefail

PORT="${1:-3005}"
N="${2:-8}"
URL="http://127.0.0.1:${PORT}"
QUERY='{"query":"Who wrote The Lord of the Rings?","top_n":3,"return_score":false}'

now() { python3 -c 'import time; print(f"{time.time():.3f}")'; }

echo "[smoke] /health on $URL:"
if ! curl -sSf "$URL/health" | head -c 200; then
  echo
  echo "[smoke] /health FAILED — retriever not up on port $PORT" >&2
  exit 1
fi
echo
echo

echo "[smoke] sample /search response (one document shown):"
curl -sS -X POST "$URL/search" -H 'Content-Type: application/json' -d "$QUERY" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d[0] if isinstance(d,list) else d, indent=2)[:400])'
echo

# Warm: first call may JIT-compile / page in.
curl -sS -X POST "$URL/search" -H 'Content-Type: application/json' -d "$QUERY" > /dev/null

# Sequential N requests.
t0=$(now)
for ((i=0; i<N; i++)); do
  curl -sS -X POST "$URL/search" -H 'Content-Type: application/json' -d "$QUERY" > /dev/null
done
t1=$(now)
seq_t=$(python3 -c "print(f'{$t1 - $t0:.3f}')")
printf '[smoke] sequential %dx: %ss\n' "$N" "$seq_t"

# Parallel N requests.
t0=$(now)
for ((i=0; i<N; i++)); do
  curl -sS -X POST "$URL/search" -H 'Content-Type: application/json' -d "$QUERY" > /dev/null &
done
wait
t1=$(now)
par_t=$(python3 -c "print(f'{$t1 - $t0:.3f}')")
printf '[smoke] parallel   %dx: %ss\n' "$N" "$par_t"

speedup=$(python3 -c "print(f'{$seq_t / $par_t:.2f}')")
echo "[smoke] parallel speedup: ${speedup}×"

# Decide.
verdict=$(python3 -c "
sp = $seq_t / $par_t
if sp >= 3:    print('PASS')
elif sp >= 2:  print('OK')
else:          print('FAIL')
")
case "$verdict" in
  PASS) echo "[smoke] PASS — async fix is working (≥3× speedup)" ;;
  OK)   echo "[smoke] OK — modest parallelism (~2-3× speedup); acceptable but inspect for OMP contention" ;;
  FAIL)
    echo "[smoke] FAIL — parallel ≈ sequential. Async fix not active." >&2
    echo "[smoke]   verify: grep -c '^async def search' local_retriever/retriever_serving.py  (expect 0)" >&2
    exit 2
    ;;
esac

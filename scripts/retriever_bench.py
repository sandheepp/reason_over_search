"""Benchmark a running retriever_serving.py instance.

Sends warmup + measurement queries (single + batch), records per-request
latency, and samples host CPU / RSS / GPU util / VRAM throughout. Writes a
JSON report at --out.

Usage (server must already be running on --port):
    python scripts/retriever_bench.py --label cpu_ivf --out cpu_ivf.json
    python scripts/retriever_bench.py --label gpu_ivf --out gpu_ivf.json --pid <retriever_pid>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any

# Hand-picked Wikipedia-style queries. Mix of named entities, multi-hop, and
# generic factoid prompts so we don't accidentally hit a single FAISS hot-path.
QUERIES = [
    "Who wrote The Lord of the Rings?",
    "What is the capital of France?",
    "Largest planet in our solar system",
    "When did World War II end?",
    "Who painted the Mona Lisa?",
    "What is the speed of light in a vacuum?",
    "Who developed the theory of general relativity?",
    "What language is spoken in Brazil?",
    "Who founded Microsoft?",
    "What is the longest river in the world?",
    "Who wrote Hamlet?",
    "When was the Eiffel Tower built?",
    "Who is the author of 1984?",
    "What element has the chemical symbol Au?",
    "Who composed the Ninth Symphony?",
    "What is the boiling point of water in Celsius?",
    "Who was the first person on the Moon?",
    "What is the smallest country in the world?",
    "Who painted the ceiling of the Sistine Chapel?",
    "What year did the Titanic sink?",
    "Who invented the telephone?",
    "What is the capital of Australia?",
    "When was the Berlin Wall built?",
    "Who wrote Pride and Prejudice?",
    "What is the deepest ocean on Earth?",
    "Who discovered penicillin?",
    "What is the tallest mountain in Africa?",
    "Who was the 16th president of the United States?",
    "When did the Roman Empire fall?",
    "Who wrote the Iliad?",
    "What is the most spoken language in the world?",
    "Who painted Starry Night?",
]


def http_post(url: str, payload: dict, timeout: float = 120.0) -> tuple[float, int, bytes]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        status = resp.status
    return time.perf_counter() - t0, status, body


def http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read()


def wait_for_health(base_url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status, _ = http_get(f"{base_url}/health", timeout=2.0)
            if status == 200:
                return True
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(2.0)
    return False


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = max(0, min(len(xs_sorted) - 1, int(round((p / 100.0) * (len(xs_sorted) - 1)))))
    return xs_sorted[k]


def stats_block(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {}
    return {
        "n": len(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p90_ms": percentile(latencies_ms, 90),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "stdev_ms": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
    }


def gpu_snapshot() -> dict[str, Any] | None:
    """Return GPU0 utilisation + VRAM via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
                "-i", "0",
            ],
            text=True, timeout=2.0,
        ).strip()
        util, mem_used, mem_total, power = [s.strip() for s in out.split(",")]
        return {
            "gpu_util_pct": float(util),
            "vram_used_mib": float(mem_used),
            "vram_total_mib": float(mem_total),
            "power_w": float(power),
        }
    except Exception:
        return None


def proc_snapshot(pid: int | None) -> dict[str, Any] | None:
    """RSS + CPU% for a specific process tree (parent + children) via ps."""
    if pid is None or not os.path.exists(f"/proc/{pid}"):
        return None
    try:
        # All descendants
        out = subprocess.check_output(
            ["ps", "-o", "pid,pcpu,rss", "--ppid", str(pid), "--pid", str(pid), "--no-headers"],
            text=True, timeout=2.0,
        )
        total_cpu = 0.0
        total_rss_kib = 0
        nproc = 0
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                total_cpu += float(parts[1])
                total_rss_kib += int(parts[2])
                nproc += 1
            except ValueError:
                continue
        return {
            "proc_count": nproc,
            "cpu_pct_sum": total_cpu,
            "rss_mib": total_rss_kib / 1024.0,
        }
    except Exception:
        return None


class Sampler(threading.Thread):
    """Background sampler: every `interval_s`, snapshot GPU + process stats."""

    def __init__(self, pid: int | None, interval_s: float = 0.5):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            ts = time.time()
            self.samples.append({
                "t": ts,
                "gpu": gpu_snapshot(),
                "proc": proc_snapshot(self.pid),
            })
            self._stop_evt.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_evt.set()

    def aggregate(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        gpu_utils = [s["gpu"]["gpu_util_pct"] for s in self.samples if s["gpu"]]
        vram = [s["gpu"]["vram_used_mib"] for s in self.samples if s["gpu"]]
        power = [s["gpu"]["power_w"] for s in self.samples if s["gpu"]]
        cpu = [s["proc"]["cpu_pct_sum"] for s in self.samples if s["proc"]]
        rss = [s["proc"]["rss_mib"] for s in self.samples if s["proc"]]
        def summary(xs: list[float]) -> dict[str, float]:
            if not xs:
                return {}
            return {
                "mean": statistics.fmean(xs),
                "max": max(xs),
                "min": min(xs),
                "p50": percentile(xs, 50),
                "p95": percentile(xs, 95),
            }
        return {
            "samples_taken": len(self.samples),
            "interval_s": self.interval_s,
            "gpu_util_pct": summary(gpu_utils),
            "vram_used_mib": summary(vram),
            "power_w": summary(power),
            "proc_cpu_pct_sum": summary(cpu),
            "proc_rss_mib": summary(rss),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3005)
    parser.add_argument("--label", required=True, help="Tag for the run, e.g. 'cpu_ivf'")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--pid", type=int, default=None,
                        help="Server PID for CPU/RSS sampling (root pid; children scanned)")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--single-iters", type=int, default=64,
                        help="Number of single-query /search calls to time")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-iters", type=int, default=16)
    parser.add_argument("--health-timeout", type=float, default=180.0)
    parser.add_argument("--sampler-interval", type=float, default=0.5)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"[{args.label}] waiting for health at {base}...")
    if not wait_for_health(base, args.health_timeout):
        print(f"[{args.label}] ERROR: server not healthy after {args.health_timeout}s")
        return 2
    print(f"[{args.label}] server healthy")

    sampler = Sampler(args.pid, args.sampler_interval)
    sampler.start()
    t_run_start = time.time()

    # Warmup
    print(f"[{args.label}] warmup x{args.warmup}")
    for i in range(args.warmup):
        q = QUERIES[i % len(QUERIES)]
        http_post(f"{base}/search", {"query": q, "top_n": args.top_n, "return_score": False})

    # Single-query latency
    print(f"[{args.label}] single-query x{args.single_iters}")
    single_lat: list[float] = []
    t0 = time.perf_counter()
    for i in range(args.single_iters):
        q = QUERIES[i % len(QUERIES)]
        dt, status, _ = http_post(
            f"{base}/search",
            {"query": q, "top_n": args.top_n, "return_score": False},
        )
        if status != 200:
            print(f"  WARN: status={status}")
        single_lat.append(dt * 1000.0)
    single_wall = time.perf_counter() - t0

    # Batch-query latency
    print(f"[{args.label}] batch x{args.batch_iters} (size {args.batch_size})")
    batch_lat: list[float] = []
    t0 = time.perf_counter()
    for i in range(args.batch_iters):
        batch = [QUERIES[(i * args.batch_size + j) % len(QUERIES)]
                 for j in range(args.batch_size)]
        dt, status, _ = http_post(
            f"{base}/batch_search",
            {"query": batch, "top_n": args.top_n, "return_score": False},
        )
        if status != 200:
            print(f"  WARN: status={status}")
        batch_lat.append(dt * 1000.0)
    batch_wall = time.perf_counter() - t0

    sampler.stop()
    sampler.join(timeout=2.0)

    qps_single = args.single_iters / single_wall if single_wall > 0 else 0.0
    qps_batch_eff = (args.batch_iters * args.batch_size) / batch_wall if batch_wall > 0 else 0.0

    report = {
        "label": args.label,
        "host": args.host,
        "port": args.port,
        "pid": args.pid,
        "top_n": args.top_n,
        "warmup": args.warmup,
        "duration_s": time.time() - t_run_start,
        "single": {
            "iters": args.single_iters,
            "wall_s": single_wall,
            "qps": qps_single,
            "latency_ms": stats_block(single_lat),
        },
        "batch": {
            "iters": args.batch_iters,
            "size": args.batch_size,
            "wall_s": batch_wall,
            "effective_qps": qps_batch_eff,
            "request_latency_ms": stats_block(batch_lat),
            "per_item_latency_ms": stats_block([x / args.batch_size for x in batch_lat]),
        },
        "system": sampler.aggregate(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{args.label}] wrote {args.out}")

    s = report["single"]["latency_ms"]
    b = report["batch"]["request_latency_ms"]
    print(f"[{args.label}] single mean={s.get('mean_ms', 0):.1f} ms  "
          f"p50={s.get('p50_ms', 0):.1f}  p95={s.get('p95_ms', 0):.1f}  "
          f"qps={qps_single:.2f}")
    print(f"[{args.label}] batch  mean={b.get('mean_ms', 0):.1f} ms  "
          f"p50={b.get('p50_ms', 0):.1f}  p95={b.get('p95_ms', 0):.1f}  "
          f"effective_qps={qps_batch_eff:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

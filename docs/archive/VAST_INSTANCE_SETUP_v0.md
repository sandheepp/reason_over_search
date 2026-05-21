---
title: VAST INSTANCE SETUP (v0, archived)
tags: [archive, setup, vast, m1]
source: internal
created: 2026-05-01
updated: 2026-05-02
archived: 2026-05-09
---

> **ARCHIVED 2026-05-09.** This is the original M1-era manual-download walkthrough.
> Successor: [`docs/vast/SETUP_VAST.md`](../vast/SETUP_VAST.md) — bootstrap-script-based
> total setup (retrieval + M2 training + M4 Qwen3.5-0.8B eval) using
> [`training/scripts/bootstrap.sh`](../../training/scripts/bootstrap.sh).
>
> What changed:
> - Manual `huggingface-cli download` steps replaced by one-shot `bootstrap.sh` (idempotent).
> - Default retriever index switched from flat IP (~65 GB, paper-fidelity) to IVF-SQ8 (~16 GB); flat IP is no longer used.
> - Scope expanded from M1 eval-only to also cover M2 training + M4 eval-models setup.
> - Search-R1 GRPO checkpoint downloads (Qwen2.5-3B, M1) dropped; replaced by Qwen3.5-0.8B (M4).
>
> Kept here for historical reference (Plan A reproduction record); not maintained.

# Vast.ai Instance Setup — From Zero to Running Eval (v0, archived)

End-to-end walkthrough to bring up a new Vast.ai instance with the corpus, indexes, encoder, GRPO checkpoints, and eval datasets all staged. After this, the box can serve the retriever, run SGLang, and execute `scripts/run_one.sh` for any (variant, dataset, seed) combo.

For *which* Vast configuration to pick for cost-optimal Plan A see [VAST_AI_PLAN_A.md](../setup/VAST_AI_PLAN_A.md). For *what* the eval pipeline does see [../milestone_1/MILESTONE_1.md](../milestone_1/MILESTONE_1.md). For ops once it's running see [../eval/EVAL_OPS.md](../eval/EVAL_OPS.md).

## 1. Pick the instance

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| GPU VRAM | 24 GB | 24–80 GB | SGLang 3B bf16 = ~22 GB |
| GPU model | RTX 4090 / 5090 / A100 / H100 | H100 PCIe for Plan A speed | See [HARDWARE_4090.md](../setup/HARDWARE_4090.md) |
| Disk | 120 GB | **150 GB** | corpus 14 + indexes 76 + 2 GRPOs 27 + eval data ~1 + headroom |
| Host RAM | 80 GB | 128 GB+ | Flat IP index alone needs ~65 GB resident; corpus mmap adds ~14 GB |
| CPU | any modern x86_64 | 16+ vCPUs | Multi-worker FAISS scaling |

**For Plan B reproduction or benchmarking**: any 24 GB GPU + 80 GB RAM is fine.
**For Plan A on this single box**: see [VAST_AI_PLAN_A.md Option 2](../setup/VAST_AI_PLAN_A.md) — single H100 PCIe per instance, 3 instances total.

## 2. Launch from the image

Vast template fields:

| Field | Value |
|---|---|
| Image | `pantomiman/reason-over-search-v1:v1` |
| Disk space | 150 GB (or whatever you sized in step 1) |
| Launch mode | Use the default Vast SSH/Jupyter mode |
| On-start script | leave blank — we'll do downloads manually |

The image already contains:
- `retriever` conda env (Python 3.10, FAISS-CPU, FastAPI, etc.)
- `evaluation_search_r1` conda env (Python 3.11, SGLang, FlashRAG)
- App code at `/app/local_retriever` and `/app/evaluation_search_r1`
- A boot hook at `/etc/vast_boot.d/10-fix-ssh-perms.sh` that fixes `/root/.ssh` perms before sshd starts (avoids "bad ownership or modes for file" errors)

What it does **not** contain (mount or download): the corpus, indexes, encoder, GRPO checkpoints, and eval datasets — too big to bake into the image. We download them in step 4.

## 3. Connect

Once the instance shows "Running" in the Vast UI:

```bash
# Vast gives you the line; format is:
ssh -o StrictHostKeyChecking=accept-new -p <PORT> root@ssh<HOST>.vast.ai
```

If you see `bad ownership or modes for file /root/.ssh/authorized_keys`, the boot hook didn't run (rare). Manual fix once inside via Vast's web shell:

```bash
chown root:root /root/.ssh /root/.ssh/authorized_keys
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
```

## 4. Download all assets

The app code is at `/app`, but it's a read-only-ish layer baked into the image. Stage a working copy under `/workspace` (Vast's persistent volume mount point) so downloads survive image rebuilds:

```bash
cd /workspace
git clone https://github.com/<your-org>/reason_over_search.git
cd reason_over_search
git checkout eval_final   # or whichever branch is locked

# Use the conda envs from the image
source /opt/miniforge3/etc/profile.d/conda.sh
```

### 4a. Corpus + Flat IP index (required) + IVF-SQ8 (optional)

```bash
conda activate retriever
cd /workspace/reason_over_search/local_retriever
mkdir -p corpus indexes models
pip install -U "huggingface_hub[cli]"   # if not already in the env

# Corpus — ~13 GB compressed → ~14 GB uncompressed
huggingface-cli download PeterJinGo/wiki-18-corpus \
  --repo-type dataset \
  --include "wiki-18.jsonl.gz" \
  --local-dir corpus --local-dir-use-symlinks False
gunzip -f corpus/wiki-18.jsonl.gz
mv corpus/wiki-18.jsonl corpus/wiki18_100w.jsonl

# Flat IP index — split into part_aa / part_ab on HF
huggingface-cli download PeterJinGo/wiki-18-e5-index \
  --repo-type dataset \
  --local-dir indexes --local-dir-use-symlinks False
cat indexes/part_aa indexes/part_ab > indexes/wiki18_100w_e5_flat_inner.index.gz
gunzip -f indexes/wiki18_100w_e5_flat_inner.index.gz
rm -f indexes/part_aa indexes/part_ab
```

**Optional: IVF-SQ8 index** (faster, lower RAM — see [../retriever/RETRIEVER_INDEXING.md](../retriever/RETRIEVER_INDEXING.md)). Download the pre-built one from HF:

```bash
cd /workspace/reason_over_search
hf download pantomiman/reason-over-search retriever/wiki18_100w_e5_ivf4096_sq8.index \
  --repo-type dataset --local-dir local_retriever/indexes
mv local_retriever/indexes/retriever/wiki18_100w_e5_ivf4096_sq8.index local_retriever/indexes/
rmdir local_retriever/indexes/retriever
```

### 4b. E5-base-v2 encoder (~0.5 GB)

```bash
cd /workspace/reason_over_search/local_retriever
huggingface-cli download intfloat/e5-base-v2 \
  --local-dir models/e5-base-v2 --local-dir-use-symlinks False
```

### 4c. GRPO checkpoints — base + instruct (~27 GB total)

```bash
cd /workspace/reason_over_search/evaluation_search_r1

huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo \
  --local-dir search_r1_base_model --local-dir-use-symlinks False

huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo \
  --local-dir search_r1_instruct_model --local-dir-use-symlinks False
```

**Verify sha256s** against [../eval/REPRODUCIBILITY.md](../eval/REPRODUCIBILITY.md#models--confirmed-grpo). A wrong-download (raw Qwen instead of GRPO) will silently produce wrong eval numbers.

```bash
# Shard 1 of base GRPO should be 7ac54e1b…36a9dabf
sha256sum search_r1_base_model/model-00001-of-00003.safetensors | head -c 8 ; echo
# Expect: 7ac54e1b
```

### 4d. Eval datasets (already in repo)

Eval splits **ship in the repo** under `data/<dataset>/<split>.jsonl` (~119 MB, full splits) and `data_subsample/<dataset>/<split>.jsonl` (~18 MB, deterministic 1 k subsamples used by the Plan B v1 sweep). A normal clone has them — no download step needed. Source: [`RUC-NLPIR/FlashRAG_datasets`](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

Confirm row counts match [../eval/PAPER_VS_OURS_AUDIT.md §G](../eval/PAPER_VS_OURS_AUDIT.md#g-datasets--splits):

```bash
for ds in nq:test triviaqa:test popqa:test hotpotqa:dev \
          2wikimultihopqa:dev musique:dev bamboogle:test; do
  d="${ds%:*}"; s="${ds#*:}"
  n=$(wc -l < "data/$d/$s.jsonl" 2>/dev/null || echo MISSING)
  echo "$d/$s.jsonl: $n"
done
# Expected: nq=3610, triviaqa=11313, popqa=14267, hotpotqa=7405,
#           2wikimultihopqa=12576, musique=2417, bamboogle=125
```

If `data/` is missing for some reason, re-pull:

```bash
cd /workspace/reason_over_search
hf download RUC-NLPIR/FlashRAG_datasets --repo-type dataset \
  bamboogle/test.jsonl nq/test.jsonl triviaqa/test.jsonl popqa/test.jsonl \
  hotpotqa/dev.jsonl 2wikimultihopqa/dev.jsonl musique/dev.jsonl \
  --local-dir data
```

Subsamples (`data_subsample/`) are also tracked. To rebuild from scratch:

```bash
scripts/subsample.sh
# Creates data_subsample/ with 1k-row deterministic samples for the large datasets
```

## 5. Smoke test

### 5a. Retriever (port 3005)

```bash
cd /workspace/reason_over_search/local_retriever
conda activate retriever

# CPU + Flat IP — safe default
nohup python retriever_serving.py \
  --config retriever_config.yaml \
  --num_retriever 2 \
  --port 3005 \
  > /tmp/retriever.log 2>&1 &
disown

# Wait for ready (FAISS load takes 30–90 s on first cold-cache read)
until curl -sS http://127.0.0.1:3005/health 2>/dev/null | grep -q healthy; do sleep 5; done
echo "retriever ready"

# Functional check
curl -sS -X POST http://127.0.0.1:3005/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"who wrote The Lord of the Rings?","top_n":3}' | head -c 400
```

Expect 3 documents, the top result mentioning Tolkien.

### 5b. SGLang (port 3000)

```bash
cd /workspace/reason_over_search
scripts/manage_sglang.sh switch instruct   # or 'base'
# Blocks until /get_model_info responds (up to ~10 min for first model load)
```

Functional check:

```bash
curl -sS http://127.0.0.1:3000/get_model_info | head -c 200
```

### 5c. Run one eval (1 dataset, 1 seed, ~5–10 min)

```bash
cd /workspace/reason_over_search
conda activate evaluation_search_r1
scripts/run_one.sh instruct bamboogle 1
```

Expected EM around **0.36** for instruct/bamboogle ([../report/RESULTS_m1.md](../report/RESULTS_m1.md)). If it's wildly off, the GRPO checkpoint is the wrong file or the prompt template drifted — re-check sha256s and `apply_chat=True`.

## 6. Disk usage check after staging

```bash
du -sh /workspace/reason_over_search/{data,local_retriever/{corpus,indexes,models},evaluation_search_r1/search_r1_*_model}
```

Expected:

```
~1 GB     data/
~14 GB    local_retriever/corpus/
~65 GB    local_retriever/indexes/        (flat only)
~81 GB    local_retriever/indexes/        (flat + IVF-SQ8)
~0.5 GB   local_retriever/models/
~14 GB    evaluation_search_r1/search_r1_base_model/
~14 GB    evaluation_search_r1/search_r1_instruct_model/
─────
~108 GB total (flat only) / ~124 GB (both indexes)
```

## 7. Background process discipline

Vast SSH connections occasionally drop. Use `nohup … & disown` for anything long-running, write logs to `/tmp/` or `/workspace/logs/`. Survives the SSH drop.

```bash
mkdir -p /workspace/logs
nohup scripts/sweep_b_reduced.sh > /workspace/logs/sweep_b.log 2>&1 &
disown
tail -f /workspace/logs/sweep_b.log
```

Or use `tmux`:

```bash
tmux new -s eval
# inside tmux: run anything
# Ctrl-b d   — detach
# tmux attach -t eval   — reattach later
```

## 8. Tear-down checklist (before destroying the instance)

If results matter:

```bash
# Pull results to your laptop / S3 / R2
rclone sync /workspace/reason_over_search/evaluation_search_r1/results \
            r2:reason-over-search/results-$(date +%F)

# Or
tar czf /tmp/results-$(date +%F).tar.gz \
  /workspace/reason_over_search/evaluation_search_r1/results
scp -P <PORT> root@ssh<HOST>.vast.ai:/tmp/results-*.tar.gz .
```

Then destroy the instance from the Vast UI. The persistent volume billing stops on destroy.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OSError: model.safetensors not found` from SGLang | GRPO download incomplete | Re-run `huggingface-cli download` for that model; `du -sh search_r1_*_model` should be ~14 GB each |
| Retriever takes >5 min to load | First-time cold-cache read of 65 GB flat index from disk | Normal on first start; subsequent starts use page cache |
| `EM` wildly off paper numbers | Wrong checkpoint, or `apply_chat=False` for base | Verify sha256 vs [../eval/REPRODUCIBILITY.md](../eval/REPRODUCIBILITY.md); check [scripts/run_one.sh:35](../../scripts/run_one.sh#L35) |
| `bad ownership or modes for file /root/.ssh/authorized_keys` | Boot hook didn't run | See step 3 manual fix |
| `cuda out of memory` from SGLang | Another process holds VRAM | `nvidia-smi` to find culprit; `pkill -f <name>` |
| Retriever RSS grows linearly with `--num_retriever` | The duplication issue from [../retriever/RETRIEVER_CONCURRENCY.md](../retriever/RETRIEVER_CONCURRENCY.md) | Apply the `IO_FLAG_MMAP` fix or use IVF-SQ8 (smaller index) |

## Quick reference — paths after setup

```
/workspace/reason_over_search/
├── data/                                         # 7 eval datasets
├── data_subsample/                               # Plan B subsampled (optional)
├── local_retriever/
│   ├── corpus/wiki18_100w.jsonl                  # 14 GB
│   ├── indexes/wiki18_100w_e5_ivf4096_sq8.index  # 16 GB (default; HF: pantomiman/reason-over-search)
│   ├── indexes/wiki18_100w_e5_flat_inner.index   # 65 GB (Flat IP; optional, exact recall)
│   └── models/e5-base-v2/                        # 0.5 GB encoder
└── evaluation_search_r1/
    ├── search_r1_base_model/                     # 14 GB GRPO base
    ├── search_r1_instruct_model/                 # 14 GB GRPO instruct
    └── results/                                  # eval outputs land here
```

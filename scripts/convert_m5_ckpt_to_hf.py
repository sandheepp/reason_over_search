#!/usr/bin/env python3
"""Convert a NeMo-RL consolidated checkpoint to HF-loadable format for SGLang.

Strategy: take the base Qwen3.5-0.8B HF model dir as a template (it has the
right config.json + tokenizer_config.json + chat_template.jinja + an HF-named
safetensors file), then overlay our trained shard's tensors over base's. The
15 mtp.* keys missing from our shard are kept from base (irrelevant for normal
inference; only used for speculative decoding training).

Usage:
    python convert_m5_ckpt_to_hf.py \\
        --ckpt-dir <step_N dir from HF download> \\
        --base-dir <Qwen3.5-0.8B HF dir> \\
        --out-dir  <output HF dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _load_shard(path: Path) -> dict:
    state = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="step_N/ from HF download")
    ap.add_argument("--base-dir", required=True, help="base Qwen3.5-0.8B HF dir")
    ap.add_argument("--out-dir", required=True, help="output HF dir for SGLang")
    args = ap.parse_args()

    ckpt = Path(args.ckpt_dir)
    base = Path(args.base_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Copy non-weight files from base (config + tokenizer + generation_config etc.)
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out / f.name)
    print(f"[copy] base meta files -> {out}")

    # 2. Copy tokenizer files from ckpt's policy/tokenizer/ to override base's
    #    (the trained tokenizer matches base but contains any post-training tweaks).
    ckpt_tok = ckpt / "policy" / "tokenizer"
    if ckpt_tok.is_dir():
        for f in ckpt_tok.iterdir():
            if f.is_file():
                shutil.copy2(f, out / f.name)
        print(f"[copy] ckpt tokenizer files -> {out}")

    # 3. Load base safetensors (488 keys: lang + visual + mtp).
    base_st = sorted([p for p in base.glob("*.safetensors*") if not str(p).endswith(".json")])[0]
    print(f"[load] base safetensors: {base_st.name}")
    base_state = _load_shard(base_st)

    # 4. Load our trained weights (473 keys: lang + visual, no mtp).
    #    DP/FSDP runs save BOTH per-rank shard-000NN-*.safetensors (each PARTIAL;
    #    reading one alone yields a broken model) AND a merged full copy under
    #    model/consolidated/. Always prefer the consolidated full model; fall
    #    back to a single shard only for legacy single-rank ckpts.
    ckpt_st = ckpt / "policy" / "weights" / "model"
    consolidated = ckpt_st / "consolidated"
    cons_files = sorted(consolidated.glob("model-*.safetensors")) if consolidated.is_dir() else []
    if cons_files:
        print(f"[load] consolidated full model: {len(cons_files)} file(s)")
        our_state = {}
        for f in cons_files:
            our_state.update(_load_shard(f))
    else:
        shard = next(ckpt_st.glob("shard-*.safetensors"))
        print(f"[load] legacy single shard: {shard.name}")
        our_state = _load_shard(shard)

    # 5. Overlay: replace every key in base_state that we have a trained version of.
    n_replaced, n_missing = 0, 0
    for k, v in our_state.items():
        if k in base_state:
            base_state[k] = v
            n_replaced += 1
        else:
            n_missing += 1
            print(f"  [warn] our key not in base: {k}")
    n_kept = len(base_state) - n_replaced
    print(f"[merge] replaced={n_replaced} (from trained)  kept={n_kept} (from base, mtp.*)  unknown={n_missing}")

    # 6. Save merged state dict. Use the base shard's filename so the existing
    #    model.safetensors.index.json (also copied) still resolves.
    out_st = out / base_st.name
    save_file(base_state, str(out_st), metadata={"format": "pt"})
    print(f"[save] {out_st} ({out_st.stat().st_size / 1e9:.2f} GB)")

    print(f"[done] HF-loadable checkpoint at {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SGLang -> vLLM HTTP protocol proxy.

flashrag/generator/generator.py:sgl_remote uses SGLang native /generate:
    POST /generate
    {"text": ["..."], "sampling_params": {"temperature":0,"max_new_tokens":N,"stop":[...]}}
    -> {"text":"...", "meta_info":{"finish_reason":{"matched":...}}}

vLLM 0.17 OpenAI /v1/completions:
    POST /v1/completions
    {"model":"...", "prompt":"...", "max_tokens":N, "temperature":0, "stop":[...]}
    -> {"choices":[{"text":"...","finish_reason":"stop|length|..."}]}

This proxy translates each direction so we can use vLLM as a drop-in backend
for the existing eval pipeline (sgl_kernel has no sm120 prebuilt for
Blackwell, so SGLang server fails to start; vLLM works fine).

Usage:
    python sgl_to_vllm_proxy.py --proxy-port 3010 --vllm-url http://127.0.0.1:3000 --model /path/to/hf_ckpt
"""
import argparse
import asyncio
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
import uvicorn


def make_app(vllm_url: str, model_name: str) -> FastAPI:
    app = FastAPI()
    base = vllm_url.rstrip("/")
    completions_url = base + "/v1/completions"
    models_url = base + "/v1/models"

    async def _one(text: str, sp: Dict[str, Any], client: httpx.AsyncClient, want_logprobs: bool) -> Dict[str, Any]:
        vllm_req: Dict[str, Any] = {
            "model": model_name,
            "prompt": text,
            "max_tokens": int(sp.get("max_new_tokens", sp.get("max_tokens", 1024))),
            "temperature": float(sp.get("temperature", 0.0)),
        }
        if "top_p" in sp:
            vllm_req["top_p"] = float(sp["top_p"])
        if "stop" in sp and sp["stop"]:
            vllm_req["stop"] = sp["stop"]
        if want_logprobs:
            vllm_req["logprobs"] = 5
        r = await client.post(completions_url, json=vllm_req, timeout=httpx.Timeout(30.0, read=1800.0))
        if r.status_code >= 400:
            raise HTTPException(r.status_code, "vLLM error: " + r.text[:1500])
        j = r.json()
        ch = j["choices"][0]
        finish = ch.get("finish_reason")
        # vLLM exposes the matched stop string (or token id) in `stop_reason` on the
        # choice when finish_reason == "stop"; SGLang puts the same info into
        # `meta_info.finish_reason.matched`, which the flashrag eval pipeline reads
        # to decide whether to invoke the retriever (`</tool_call>`, `</search>`),
        # treat as final answer (`</answer>`, `<|im_end|>`, `<|endoftext|>`), or
        # error out. None breaks flashrag's `isinstance(stop_matched, str)` branch.
        matched: Any = ch.get("stop_reason")
        # Early-step ckpts (step_10..40 of training; near-untrained model) often
        # finish on a natural EOS token rather than a configured stop-string, so
        # vLLM returns finish_reason="stop" with stop_reason=None. flashrag then
        # falls through every branch and raises
        #   ValueError: stop_reason: stop, stop_matched: None
        # killing the eval. Map (stop, None) -> "<|im_end|>" which flashrag accepts
        # as an EOS sentinel (active_pipeline.py:296-302 includes it in the str set).
        if matched is None and (finish or "stop") == "stop":
            matched = "<|im_end|>"
        # vLLM sometimes strips the matched stop string from `text`; flashrag
        # tolerates either (it re-appends action_stop when missing).
        return {
            "text": ch.get("text", ""),
            "meta_info": {
                "finish_reason": {
                    "matched": matched,
                    "type": finish or "stop",
                },
                "prompt_tokens": (j.get("usage") or {}).get("prompt_tokens", 0),
                "completion_tokens": (j.get("usage") or {}).get("completion_tokens", 0),
            },
        }

    @app.get("/health")
    async def health() -> Dict[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                r = await client.get(models_url, timeout=5.0)
                if r.status_code == 200:
                    return {"status": "healthy"}
            except Exception:
                pass
        raise HTTPException(503, "vllm backend not reachable")

    @app.post("/generate")
    async def generate(req: Request):
        body = await req.json()
        texts: List[str] = body.get("text") or []
        if isinstance(texts, str):
            texts = [texts]
        sp: Dict[str, Any] = body.get("sampling_params") or {}
        want_logprobs = bool(body.get("return_logprob", False))

        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(*[_one(t, sp, client, want_logprobs) for t in texts])
        return results if len(results) > 1 else results[0]

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-port", type=int, default=3010)
    ap.add_argument("--vllm-url", required=True)
    ap.add_argument("--model", required=True, help="HF model path; passed in vLLM requests as model_name")
    args = ap.parse_args()
    app = make_app(args.vllm_url, args.model)
    uvicorn.run(app, host="127.0.0.1", port=args.proxy_port, log_level="warning")


if __name__ == "__main__":
    main()

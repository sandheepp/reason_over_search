from fastapi import FastAPI, HTTPException
import argparse
from pydantic import BaseModel
from typing import List, Tuple, Union
import asyncio
from collections import deque

from flashrag.config import Config
from flashrag.utils import get_retriever

app = FastAPI()

retriever_list = []
available_retrievers = deque()
retriever_semaphore = None


def init_retriever(args):
    global retriever_semaphore
    config = Config(args.config)
    # CLI overrides win over both yaml and Config._init_device's hardware autodetection.
    config['faiss_gpu'] = bool(args.gpu)
    if args.index is not None:
        config['index_path'] = args.index
    print(
        f"index_path={config['index_path']}  faiss_gpu={config['faiss_gpu']}  "
        f"nprobe={config.get('faiss_nprobe')}"
    )
    for i in range(args.num_retriever):
        print(f"Initializing retriever {i+1}/{args.num_retriever}")
        retriever = get_retriever(config)
        retriever_list.append(retriever)
        available_retrievers.append(i)
    # create a semaphore to limit the number of retrievers that can be used concurrently
    retriever_semaphore = asyncio.Semaphore(args.num_retriever)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(retriever_list),
            "available": len(available_retrievers)
        }
    }


class QueryRequest(BaseModel):
    query: str
    top_n: int = 10
    return_score: bool = False


class BatchQueryRequest(BaseModel):
    query: List[str]
    top_n: int = 10
    return_score: bool = False


class Document(BaseModel):
    id: str
    contents: str


@app.post(
    "/search",
    response_model=Union[Tuple[List[Document], List[float]], List[Document]],
)
async def search(request: QueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    if not query or not query.strip():
        print(f"Query content cannot be empty: {query}")
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            # Wrap sync FAISS search in to_thread so it doesn't block the
            # asyncio event loop. FAISS releases the GIL inside its C++ search
            # path, so concurrent threads actually run in parallel and we get
            # the full benefit of `num_retriever` instances.
            payload = await asyncio.to_thread(
                retriever_list[retriever_idx].search, query, top_n, return_score
            )
            if return_score:
                results, scores = payload
                docs = [
                    Document(id=r['id'], contents=r['contents'])
                    for r in results
                ]
                return docs, scores
            else:
                results = payload
                return [
                    Document(id=r['id'], contents=r['contents'])
                    for r in results
                ]
        finally:
            available_retrievers.append(retriever_idx)


@app.post(
    "/batch_search",
    response_model=Union[
        List[List[Document]],
        Tuple[List[List[Document]], List[List[float]]],
    ],
)
async def batch_search(request: BatchQueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            payload = await asyncio.to_thread(
                retriever_list[retriever_idx].batch_search, query, top_n, return_score
            )
            if return_score:
                results, scores = payload
                batched = [
                    [
                        Document(id=r['id'], contents=r['contents'])
                        for r in results[i]
                    ]
                    for i in range(len(results))
                ]
                return batched, scores
            else:
                results = payload
                return [
                    [
                        Document(id=r['id'], contents=r['contents'])
                        for r in results[i]
                    ]
                    for i in range(len(results))
                ]
        finally:
            available_retrievers.append(retriever_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="./retriever_config.yaml")
    parser.add_argument("--num_retriever", type=int, default=1)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument(
        "--gpu", action="store_true",
        help="Hold the FAISS index in VRAM. Requires a faiss-gpu build (see .venv setup).",
    )
    parser.add_argument(
        "--index", type=str, default=None,
        help="Override index_path from the yaml. e.g. ./indexes/wiki18_100w_e5_ivf4096_sq8.index",
    )
    args = parser.parse_args()

    init_retriever(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

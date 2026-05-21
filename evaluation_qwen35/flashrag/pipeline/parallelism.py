# Client-side parallelism for remote LLM / retriever HTTP calls.
# Default 32 worked under retriever-side bottleneck (8-worker FAISS pool);
# with retriever bumped to 32 workers and SGLang reporting max_running_requests=533
# on A100-80GB, 128 is verified empirically (RESULTS_SMOKE_m4 §8).
# Lower this if you see KV thrashing (mem_fraction_static<0.7 helps too) or
# queue blowups under a smaller retriever-worker pool.
INFERENCE_MAX_WORKERS = 128

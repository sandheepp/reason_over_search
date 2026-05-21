import os, sys
print("--- raw path checks ---")
for p in [
    "/workspace/reason_over_search/corpus/wiki18_100w.jsonl",
    "/workspace/reason_over_search/models/e5-base-v2",
    "/workspace/reason_over_search/indexes/wiki18_100w_e5_ivf4096_sq8.index",
]:
    print(f"isfile={os.path.isfile(p)} isdir={os.path.isdir(p)} exists={os.path.exists(p)} {p}")
print("--- Config parsing ---")
os.chdir("/workspace/reason_over_search/local_retriever")
from flashrag.config import Config
c = Config("retriever_config_smoke.yaml")
for k in ["corpus_path", "retrieval_method", "index_path"]:
    v = c[k]
    print(f"  {k}={v}  isfile={os.path.isfile(v)} isdir={os.path.isdir(v)}")
print("--- get_retriever attempt ---")
try:
    from flashrag.utils import get_retriever
    r = get_retriever(c)
    print("retriever ok:", type(r).__name__)
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")

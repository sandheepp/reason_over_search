import importlib
from transformers import AutoConfig


def get_retriever(config):
    r"""Select retriever class from ``retrieval_method`` and related flags."""
    if config.get("use_remote_retriever", False):
        return getattr(importlib.import_module("flashrag.retriever"), "RemoteRetriever")(config)

    if config.get("use_multi_retriever", False):
        return getattr(importlib.import_module("flashrag.retriever"), "MultiRetrieverRouter")(config)

    if config["retrieval_method"] == "bm25":
        return getattr(importlib.import_module("flashrag.retriever"), "BM25Retriever")(config)
    else:
        try:
            model_config = AutoConfig.from_pretrained(config["retrieval_model_path"])
            arch = model_config.architectures[0]
            if "clip" in arch.lower():
                return getattr(importlib.import_module("flashrag.retriever"), "MultiModalRetriever")(config)
            else:
                return getattr(importlib.import_module("flashrag.retriever"), "DenseRetriever")(config)
        except Exception:
            return getattr(importlib.import_module("flashrag.retriever"), "DenseRetriever")(config)


def get_reranker(config):
    model_path = config["rerank_model_path"]
    model_config = AutoConfig.from_pretrained(model_path)
    arch = model_config.architectures[0]
    if "forsequenceclassification" in arch.lower():
        return getattr(importlib.import_module("flashrag.retriever"), "CrossReranker")(config)
    else:
        return getattr(importlib.import_module("flashrag.retriever"), "BiReranker")(config)

from flashrag.config import Config
from flashrag.utils import get_dataset
import argparse


def _str2bool(v):
    """argparse bool: type=bool treats str(\"False\") as True — use this instead."""
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("yes", "true", "t", "1", "y"):
        return True
    if s in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def _dataset_for_split(all_split, split_name, config):
    """Return the Dataset for split or raise with paths (avoids opaque NoneType errors)."""
    data = all_split.get(split_name)
    if data is None:
        dp = config["dataset_path"]
        raise ValueError(
            f"No data for split {split_name!r}. dataset_path={dp!r} "
            f"(join of data_dir and dataset_name). "
            f"Expected {split_name}.jsonl, .json, or .parquet under that directory. "
            f"If files are in .../data/bamboogle/, use --data_dir .../data (not .../data/bamboogle) "
            f"with --dataset_name bamboogle."
        )
    return data


def naive(args, config_dict):
    from flashrag.pipeline import SequentialPipeline

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = _dataset_for_split(all_split, args.split, config)

    pipeline = SequentialPipeline(config)

    result = pipeline.run(test_data)

def zero_shot(args, config_dict):
    from flashrag.pipeline import SequentialPipeline
    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = _dataset_for_split(all_split, args.split, config)

    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate

    templete = PromptTemplate(
        config=config,
        system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
        user_prompt="Question: {question}",
    )
    pipeline = SequentialPipeline(config, templete)
    result = pipeline.naive_run(test_data)

def iterretgen(args, config_dict):
    """
    Reference:
        Zhihong Shao et al. "Enhancing Retrieval-Augmented Large Language Models with Iterative
                            Retrieval-Generation Synergy"
        in EMNLP Findings 2023.

        Zhangyin Feng et al. "Retrieval-Generation Synergy Augmented Large Language Models"
        in EMNLP Findings 2023.
    """
    iter_num = 3

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = _dataset_for_split(all_split, args.split, config)

    from flashrag.pipeline import IterativePipeline

    pipeline = IterativePipeline(config, iter_num=iter_num)
    result = pipeline.run(test_data)

def ircot(args, config_dict):
    """
    Reference:
        Harsh Trivedi et al. "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"
        in ACL 2023
    """
    from flashrag.pipeline import IRCOTPipeline

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = _dataset_for_split(all_split, args.split, config)
    print(config["generator_model_path"])
    pipeline = IRCOTPipeline(config, max_iter=5)

    result = pipeline.run(test_data)

def search_r1(args, config_dict):
    config_dict["metrics"] = ["em", "acc", "f1"]
    config_dict["search_r1_mode"] = True
    # All qwen3* modes (p1_basic_w_ex, p3_decide_no_ex, ...) were trained with top_n=5;
    # stay at top-3 for search_r1 paper-fidelity.
    if args.prompt_mode.startswith("qwen3"):
        config_dict["retrieval_topk"] = 5
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = _dataset_for_split(all_split, args.split, config)

    from flashrag.pipeline import SearchR1Pipeline
    pipeline = SearchR1Pipeline(config, apply_chat=args.apply_chat, enable_thinking=args.enable_thinking, prompt_mode=args.prompt_mode)
    result = pipeline.run(test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--config_path", type=str, default="./eval_config.yaml")
    parser.add_argument("--method_name", type=str, default="search-r1")
    parser.add_argument("--data_dir", type=str, default="your-data-dir")
    parser.add_argument("--dataset_name", type=str, default="bamboogle")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_dir", type=str, default="your-save-dir")
    parser.add_argument("--save_note", type=str, default='your-save-note-for-identification')
    # Defaults None so eval_config.yaml is used when a shell line is accidentally split
    # (placeholders here would override YAML and break HF / remote URLs).
    parser.add_argument("--sgl_remote_url", type=str, default=None)
    parser.add_argument("--remote_retriever_url", type=str, default=None)
    parser.add_argument("--generator_model", type=str, default=None)
    parser.add_argument("--apply_chat", type=_str2bool, default=True)
    parser.add_argument("--enable_thinking", type=_str2bool, default=False)
    parser.add_argument("--search_r1_mode", type=_str2bool, default=False)
    parser.add_argument("--prompt_mode", type=str, default='search_r1',
                        help="'search_r1' (original user-msg template), 'qwen3' / 'qwen3_p1_basic_w_ex' (M3 z7kcxfof prompt), or 'qwen3_p3_decide_no_ex' (M3.1 el6s2d2h prompt)")

    func_dict = {
        "naive": naive,
        "zero-shot": zero_shot,
        "iterretgen": iterretgen,
        "ircot": ircot,
        "search-r1": search_r1,
    }

    args = parser.parse_args()

    config_dict = {
        "data_dir": args.data_dir,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "save_dir": args.save_dir,
        "save_note": args.save_note if args.save_note else args.method_name,
        "enable_thinking": args.enable_thinking,
        "search_r1_mode": args.search_r1_mode,
    }
    if args.sgl_remote_url is not None:
        config_dict["sgl_remote_url"] = args.sgl_remote_url
        # Remote SGLang endpoint should drive generation backend.
        config_dict["framework"] = "sgl_remote"
    if args.remote_retriever_url is not None:
        config_dict["remote_retriever_url"] = args.remote_retriever_url
        # A remote retriever URL implies we should call retriever service APIs.
        config_dict["use_remote_retriever"] = True
    if args.generator_model is not None:
        config_dict["generator_model"] = args.generator_model

    func = func_dict[args.method_name]
    func(args, config_dict)

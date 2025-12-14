from .lazyenv import *


def load_dataset(dataset: DatasetEnv):
    from datasets import load_dataset as hf_load_dataset
    from datasets import Dataset, load_from_disk
    import json

    """根据 DatasetEnv 加载数据集"""
    if dataset.filetype == "hf":
        ds = hf_load_dataset(dataset.path, **dataset.tags.get("loader_args", {}))
    elif dataset.filetype == "hf_disk":
        ds = load_from_disk(dataset.path, **dataset.tags.get("loader_args", {}))
    elif dataset.filetype == "json":
        with open(dataset.path, "r") as f:
            data = json.load(f)
        ds = Dataset.from_list(data)
    else:
        raise NotImplementedError(f"Unsupported dataset filetype: {dataset.filetype}")
    return ds


def load_model(
    model: ModelEnv, tokenizer_args: dict | None = None, model_args: dict | None = None
):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer_args = dict(padding_side="left", trust_remote_code=True)
    tokenizer_args.update(model.tags.get("tokenizer_args", {}))
    tokenizer_args.update(tokenizer_args or {})
    tokenizer = AutoTokenizer.from_pretrained(model.path, **tokenizer_args)
    model_args = dict(device_map="auto", trust_remote_code=True)
    model_args.update(model.tags.get("model_args", {}))
    model_args.update(model_args or {})
    mo = AutoModelForCausalLM.from_pretrained(model.path, **model_args)
    return tokenizer, mo

from .exenv import *
from datasets import Dataset, load_from_disk
from .utils import call_function_from_file

def load_dataset(dataset: DatasetEnv)->Dataset:
    from datasets import load_dataset as hf_load_dataset
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
    elif dataset.filetype == "py":
        ds = call_function_from_file(dataset.path, "load_dataset", dataset)
    else:
        raise NotImplementedError(f"Unsupported dataset filetype: {dataset.filetype}")
    if not isinstance(ds, Dataset):
        print(f"Warning: dataset {dataset.path} is not a Dataset instance, but {type(ds)}")
    return ds # type: ignore


def load_model(
    model: ModelEnv, tokenizer_args: dict | None = None, model_args: dict | None = None
):
    return load_model_only(model, model_args), load_tokenizer_only(model, tokenizer_args)

def load_tokenizer_only(model: ModelEnv, tokenizer_args: dict | None = None):
    from transformers import AutoTokenizer

    tokenizer_args = dict(padding_side="left", trust_remote_code=True)
    tokenizer_args.update(model.tags.get("tokenizer_args", {}))
    tokenizer_args.update(tokenizer_args or {})
    print("Loading tokenizer:", model.path)
    tokenizer = AutoTokenizer.from_pretrained(model.path, **tokenizer_args)
    return tokenizer

def load_model_only(model: ModelEnv, model_args: dict | None = None):
    from transformers import AutoModelForCausalLM

    model_args = dict(device_map="auto", trust_remote_code=True)
    model_args.update(model.tags.get("model_args", {}))
    model_args.update(model_args or {})
    print("Loading model:", model.path)
    mo = AutoModelForCausalLM.from_pretrained(model.path, **model_args)
    return mo

from .exenv import *
from datasets import Dataset, load_from_disk
from .utils import call_function_from_file
import _hooks

def load_dataset(dataset: DatasetEnv) -> Dataset:
    """
    Load a dataset according to the information provided by the DatasetEnv.

    Supports loading datasets directly from huggingface, locally cached
    HuggingFace disk files (.hf_disk), json files, or dynamically via a python script.

    Args:
        dataset (DatasetEnv): The environment setting the source, path and filetype of the dataset.

    Returns:
        Dataset: The loaded HuggingFace Dataset object.
    """
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
    elif dataset.filetype == "src":
        ds = call_function_from_file(dataset.path, "load_dataset", dataset)
    else:
        raise NotImplementedError(f"Unsupported dataset filetype: {dataset.filetype}")
    if not isinstance(ds, Dataset):
        print(
            f"Warning: dataset {dataset.path} is not a Dataset instance, but {type(ds)}"
        )
    for hook_name, hook_args in dataset.tags.get("load_hooks", []):
        ds = getattr(_hooks, hook_name)(ds, **hook_args)
    return ds  # type: ignore

def load_inputs(dataset: DatasetEnv) -> list[str]:
    """
    Load input data from a dataset, using the prompt template if provided.

    Args:
        dataset (DatasetEnv): The environment setting the source, path and filetype of the dataset.

    Returns:
        list[str]: A list of input strings extracted from the dataset.
    """
    ds = load_dataset(dataset)
    if dataset.prompt_template:
        inputs = [dataset.prompt_template.format(**item) for item in ds]
    else:
        inputs = [str(item) for item in ds]
    return inputs


def load_model(
    model: ModelEnv, tokenizer_args: dict | None = None, model_args: dict | None = None
):
    """
    Load an LLM and its corresponding tokenizer based on the ModelEnv.

    Args:
        model (ModelEnv): Definition of the language model to be loaded.
        tokenizer_args (dict | None, optional): Additional arguments for the transformer tokenizer.
        model_args (dict | None, optional): Additional arguments for the transformer model.

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizer]: A pair containing the model and tokenizer instances.
    """
    return load_model_only(model, model_args), load_tokenizer_only(
        model, tokenizer_args
    )


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

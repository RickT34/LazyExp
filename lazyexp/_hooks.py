import json


def loader_llm_eval(ds, output_path:str):
    outputs = json.load(open(output_path, "r"))
    assert len(ds) == len(outputs), f"dataset length {len(ds)} does not match outputs length {len(outputs)}"
    return [{"output": output, **item} for item, output in zip(ds, outputs)]
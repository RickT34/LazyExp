import json


def loader_llm_eval(ds, output_path:str, output_field:str="output"):
    outputs = json.load(open(output_path, "r"))
    assert len(ds) == len(outputs), f"dataset length {len(ds)} does not match outputs length {len(outputs)}"
    return [{output_field: output, **item} for item, output in zip(ds, outputs)]
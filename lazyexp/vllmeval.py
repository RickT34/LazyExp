import argparse
import json
import os
from . import envloader, exenv
from datasets import Dataset
import importlib


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM High-Performance Evaluation Script (VLLM Backend)"
    )
    parser.add_argument("--env", type=str, required=True, help="env file path")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max_ctx_len",
        type=int,
        default=16384,
        help="Maximum context length (prompt + generation). Adjust based on your model's capacity.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction",
    )
    return parser.parse_args()


def main(env: exenv.ExpEnv, max_new_tokens: int=1024, max_ctx_len: int=16384, gpu_memory_utilization: float=0.9, skip_exist: bool=True):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("Please install vllm to use LLMEvaluator: pip install vllm")

    if skip_exist and os.path.exists(env.get_output_path()):
        print(f"Skipping {env}.")
        exit(0)
    # 1. Load Environment
    # env = exenv.ExpEnv.load(args.env)

    # 2. Load Dataset
    print(f"Loading dataset...")
    dataset: Dataset = envloader.load_dataset(env.dataset)
    print(f"Total samples: {len(dataset)}")

    # 3. Load Tokenizer (for chat template application)

    tokenizer = envloader.load_tokenizer_only(env.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Prepare Prompts
    assert (
        "prompt_template" in env.dataset.tags
    ), "DatasetEnv must have 'prompt_template' tag"
    prompt_template = env.dataset.tags["prompt_template"]

    print("Preparing prompts...")
    prompts = []
    model_len = max_ctx_len
    sampling_times = env.tags.get("sampling_times", 1)
    for item in dataset:
        # Format user input using data items
        user_content = prompt_template.format(**item)

        messages = []
        if "system_prompt" in env.dataset.tags:
            messages.append(
                {"role": "system", "content": env.dataset.tags["system_prompt"]}
            )
        messages.append({"role": "user", "content": user_content})

        # Apply chat template to get the final text prompt
        text_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if len(text_prompt) > model_len:
            print(
                f"Warning: Prompt length {len(text_prompt)} exceeds model max context length {model_len}."
            )
        prompts.extend([text_prompt]*sampling_times)

    # 5. Initialize VLLM LLM Engine
    print(f"Initializing VLLM with model: {env.model.path}")
    # Extract any model args if present in env, though VLLM uses its own set
    # We mainly care about the path and basic valid args
    llm = LLM(
        model=env.model.path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=model_len,
        dtype="auto",
    )

    # Configure Sampling Parameters
    # llmeval.py used do_sample=False, so we use temperature=0
    sampling_args = dict(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    if "sampling_args" in env.tags:
        print(f"Updating sampling args...")
        sampling_args.update(env.tags["sampling_args"])
    print(f"sampling args:{sampling_args}")
    sampling_params = SamplingParams(**sampling_args)

    # 6. Plugins Support
    plugin_paths = env.tags.get("eval_plugins", [])
    plugin_modules = []
    # Create a simple context object or dictionary to mimic locals() if needed,
    # but strictly speaking locals() in llmeval had 'model', 'tokenizer', 'dataset', 'env' etc.
    # We try to provide compatible names where possible.
    plugin_context = locals()

    for plugin_path in plugin_paths:
        print(f"Loading plugin: {plugin_path}")
        plugin_module = importlib.import_module(plugin_path)
        plugin_modules.append(plugin_module)
        if hasattr(plugin_module, "pre_eval"):
            print(f"Running pre_eval from {plugin_path}")
            plugin_module.pre_eval(plugin_context)

    # 7. Generation
    print(f"Starting generation for {len(prompts)} samples...")
    # VLLM handles batching and progress bar automatically
    outputs = llm.generate(prompts, sampling_params)

    # 8. Extract Results
    results = []
    # Ensure results follow the same order as inputs (VLLM usually preserves order but let's be careful if we were sorting)
    # VLLM generate returns a list in the same order as prompts.
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append(generated_text.strip())

    # 9. Post Eval Plugins
    for plugin_module in plugin_modules:
        if hasattr(plugin_module, "post_eval"):
            print(f"Running post_eval from {plugin_module.__name__}")
            plugin_module.post_eval(plugin_context)

    # 10. Save Results
    print(f"Saving results...")
    output_file = env.get_output_path()
    if os.path.exists(output_file):
        print("Warning: Output file already exists.")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print("Done!")


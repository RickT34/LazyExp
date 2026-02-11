import argparse
import json
import os
import torch
from tqdm import tqdm
from lazyexp import envloader, exenv
from datasets import Dataset
import importlib


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM High-Performance Evaluation Script"
    )
    parser.add_argument("--env", type=str, required=True, help="env file path")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference. Increase this if VRAM allows.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generation",
        choices=["generation", "activation"],
        help="Evaluation mode",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    env = exenv.ExpEnv.load(args.env)

    # 2. 加载数据集
    print(f"Loading dataset...")
    dataset: Dataset = envloader.load_dataset(env.dataset)
    # dataset = dataset.select(range(10))

    print(f"Total samples: {len(dataset)}")

    # 3. 加载模型和分词器
    print(f"Loading model...")
    # 使用 float16 或 bfloat16 加速
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, tokenizer = envloader.load_model(
        env.model,
        dict(padding_side="left", trust_remote_code=True),
        dict(
            torch_dtype=torch_dtype,
            device_map="auto",  # 自动分配到指定的显卡
            trust_remote_code=True,
            attn_implementation=(
                "flash_attention_2"
                if torch.cuda.get_device_capability()[0] >= 8
                else "eager"
            ),
        ),
    )
    device = model.device

    # 确保有 pad_token，通常将 eos_token 设为 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # 4. 准备批量数据处理
    # 提取文本数据
    assert (
        "prompt_template" in env.dataset.tags
    ), "DatasetEnv must have 'prompt_template' tag"
    prompt_template = env.dataset.tags["prompt_template"]
    input_data = [prompt_template.format(**item) for item in dataset]

    # 结果容器
    results = []

    print(f"Starting {args.mode} with batch size {args.batch_size}...")

    plugin_paths = env.tags.get("eval_plugins", [])
    plugin_modeles = []
    for plugin_path in plugin_paths:
        print(f"Loading plugin: {plugin_path}")
        plugin_module = importlib.import_module(plugin_path)
        plugin_modeles.append(plugin_module)
        if hasattr(plugin_module, "pre_eval"):
            print(f"Running pre_eval from {plugin_path}")
            plugin_module.pre_eval(locals())

    # 5. 批量推理循环
    # 使用 torch.inference_mode() 减少显存占用和计算开销
    with torch.inference_mode():
        for i in tqdm(range(0, len(input_data), args.batch_size), desc="Evaluating"):
            # 获取当前 batch 的文本
            batch_texts = input_data[i : i + args.batch_size]
            batch_formatted_prompts = []
            for p in batch_texts:
                messages = []
                if "system_prompt" in env.dataset.tags:
                    messages.append(
                        {"role": "system", "content": env.dataset.tags["system_prompt"]}
                    )
                messages.append({"role": "user", "content": p})

                # 使用 chat_template 将 list 转为 string
                # tokenize=False 表示先只转成字符串，稍后统一 tokenize 以便 batch 处理
                text_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_formatted_prompts.append(text_prompt)
            # Tokenize
            # padding=True: 填充到当前 batch 最长序列
            # truncation=True: 防止输入过长爆显存 (根据需要可选)
            inputs = tokenizer(
                batch_formatted_prompts,
                return_tensors="pt",
                padding=True,  # 填充到最长
                truncation=True,
            ).to(device)
            input_length = inputs.input_ids.shape[1]

            if args.mode == "generation":
                # generated_ids 包含 [输入 + 输出]
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                # generated_ids[:, input_length:] 即只取 input_length 之后的内容
                output_ids = generated_ids[:, input_length:]
            elif args.mode == "activation":
                output_logits = model.forward(**inputs)["logits"]
                # activation_data 的 shape 通常为 (batch_size, seq_len, hidden_size)
                generated_ids = torch.argmax(output_logits, dim=-1)
                output_ids = generated_ids[:, input_length - 1 :]
            else:
                raise NotImplementedError(f"Unsupported mode: {args.mode}")

            decoded_outputs = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )

            for j, output_text in enumerate(decoded_outputs):
                results.append(output_text.strip())

    for plugin_module in plugin_modeles:
        if hasattr(plugin_module, "post_eval"):
            print(f"Running post_eval from {plugin_module.__name__}")
            plugin_module.post_eval(locals())

    # 6. 保存结果
    print(f"Saving results...")

    with open(env.get_output_path(), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    main()

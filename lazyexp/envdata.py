from .lazyenv import *

DIR_EXP_HISTORY = Path('exp_history')
DIR_DATA = '/Data2/tangrui/trsdata'

ModelLLaMA3_8B = ModelEnv("llama3-8b", f"{DIR_DATA}/models/LLama-3-8B-Instruct", 32)
ModelQwen25_7B = ModelEnv("qwen2.5-7b", f"{DIR_DATA}/models/Qwen2.5-7B-Instruct", 28)
ModelQwen25_3B = ModelEnv("qwen2.5-3b", f"{DIR_DATA}/models/Qwen2.5-3B-Instruct", 28)
ModelQwen25_05B = ModelEnv("qwen2.5-0.5b", f"{DIR_DATA}/models/Qwen2.5-0.5B-Instruct", 28)
ModelLLaMA2_7B = ModelEnv("llama-7b", f"{DIR_DATA}/models/Llama-2-7b-chat-hf", 32)
ModelLLaMA32_3B = ModelEnv("llama3.2-3b", f"{DIR_DATA}/models/Llama-3.2-3B-Instruct", 28)




DatasetsMQCF2hop800 = list(
    DatasetEnv(f"{DIR_DATA}/dataset/mq_cf_sample800_2hop/mq_cf_sample800_2hop{i+1}.json", tags={"loc": i})
    for i in range(2)
)
DatasetCF1024 = DatasetEnv(f"{DIR_DATA}/dataset/cf_formatted_k1024.json")

DatasetOT114k = DatasetEnv(f"{DIR_DATA}/dataset/OpenThoughts-114k", filetype="hf")
DatasetOT10k = DatasetEnv(f"{DIR_DATA}/dataset/OpenThoughts-10k", filetype="hf")

DatasetTULU = DatasetEnv(f"{DIR_DATA}/dataset/tulu-3-sft-mixture", filetype="hf")
DatasetTULU10k = DatasetEnv(f"{DIR_DATA}/dataset/tulu-3-sft-mixture-10k", filetype="hf")

AlgoNULL = AlgoEnv("null", {})


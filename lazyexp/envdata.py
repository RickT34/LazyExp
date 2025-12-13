from .lazyenv import *

DIR_EXP_HISTORY = Path('exp_history')

ModelLLaMA3_8B = ModelEnv("llama3-8b", "data/models/LLama-3-8B-Instruct", 32)
ModelQwen25_7B = ModelEnv("qwen2.5-7b", "data/models/Qwen2.5-7B-Instruct", 28)
ModelQwen25_3B = ModelEnv("qwen2.5-3b", "data/models/Qwen2.5-3B-Instruct", 28)
ModelQwen25_05B = ModelEnv("qwen2.5-0.5b", "data/models/Qwen2.5-0.5B-Instruct", 28)
ModelLLaMA2_7B = ModelEnv("llama-7b", "data/models/Llama-2-7b-chat-hf", 32)
ModelLLaMA32_3B = ModelEnv("llama3.2-3b", "data/models/Llama-3.2-3B-Instruct", 28)




DatasetsMQCF2hop800 = list(
    DatasetEnv(f"data/dataset/mq_cf_sample800_2hop/mq_cf_sample800_2hop{i+1}.json", tags={"loc": i})
    for i in range(2)
)
DatasetsMQCF2hop800inv = list(
    DatasetEnv(f"data/dataset/mq_cf_sample800_2hopinv/mq_cf_sample800_2hopinv{i+1}.json", tags={"loc": i})
    for i in range(2)
)
DatasetsMQCF2hop200 = list(
    DatasetEnv(f"data/dataset/mq_cf_sample200_2hop/mq_cf_sample200_2hop{i+1}.json", tags={"loc": i})
    for i in range(2)
)
DatasetsMQCF3hop100 = list(
    DatasetEnv(f"data/dataset/mq_cf_sample100_3hop/mq_cf_sample100_3hop{i+1}.json", tags={"loc": i})
    for i in range(3)
)
DatasetMQCF2chop200 = DatasetEnv("data/dataset/mq_cf_sample200_2chop_2.json")
DatasetMQCFAllEdges = DatasetEnv("data/dataset/mq_cf_all_edges.json")
DatasetCF1000 = DatasetEnv("data/dataset/cf_formatted_k1000.json")
DatasetCF256 = DatasetEnv("data/dataset/cf_formatted_k256.json")
DatasetCF1024notest = DatasetEnv("data/dataset/cf_formatted_k1024_notest.json")
DatasetCF1024 = DatasetEnv("data/dataset/cf_formatted_k1024.json")


DatasetOT114k = DatasetEnv("data/dataset/OpenThoughts-114k", filetype="hf")
DatasetOT10k = DatasetEnv("data/dataset/OpenThoughts-10k", filetype="hf")

DatasetTULU = DatasetEnv("data/dataset/tulu-3-sft-mixture", filetype="hf")
DatasetTULU10k = DatasetEnv("data/dataset/tulu-3-sft-mixture-10k", filetype="hf")

AlgoNULL = AlgoEnv("null", {})


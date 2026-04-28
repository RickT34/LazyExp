from .exenv import *
from .exper import RUNNER_TYPE
from collections import defaultdict
from typing import Callable
from . import envloader
from typing import Any

dataset_cache: dict[str, Any] = {}

def get_dataset_cached(dataset: DatasetEnv):
    if dataset.path not in dataset_cache:
        dataset_cache[dataset.path] = envloader.load_dataset(dataset)
    ds = dataset_cache[dataset.path]
    return ds


class Evaluator:
    def __init__(self, sub_src_file:str, sub_tgt_file: str):
        self.sub_tgt_file = sub_tgt_file
        self.sub_src_file = sub_src_file

    def evaluate(self, exp_env: ExpEnv):
        raise NotImplementedError
    
    def get_src_path(self, exp_env:ExpEnv):
        path = exp_env.get_output_path(self.sub_src_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist!")
        return path
    
    def get_tgt_path(self, exp_env:ExpEnv):
        return exp_env.get_output_path(self.sub_tgt_file)

class LLMEvaluator(Evaluator):
    def __init__(self, judger: ModelEnv, prompt_template: str, runner: RUNNER_TYPE, sub_src_file:str="result.json", subdir:str="llmeval", model_output_field:str="output"):
        super().__init__(sub_src_file, os.path.join(subdir, "result.json"))
        self.subdir = subdir
        self.judger = envCopy(judger, ModelEnv)
        self.prompt_template = prompt_template
        self.runner = runner
        self.model_output_field = model_output_field

    def evaluate(self, exp_env: ExpEnv):
        path = self.get_src_path(exp_env)
        dataset_new = envCopy(exp_env.dataset, DatasetEnv)
        dataset_new.prompt_template = self.prompt_template
        dataset_new.tags["load_hooks"] = [
            ("loader_llm_eval", {"output_path": path, "output_field": self.model_output_field})
        ]
        env = ExpEnv(
            model=self.judger,
            dataset=dataset_new,
            algo=AlgoEnv("loader_llm_eval"),
            label="llmeval",
            output_dir=exp_env.get_output_path(self.subdir),
        )
        self.runner(env)

class LineCheck(Evaluator):
    def __init__(self, check_func, sub_src_file:str, sub_tgt_file:str):
        self.check_func = check_func
        super().__init__(sub_src_file, sub_tgt_file)
    
    def evaluate(self, exp_env: ExpEnv):
        path = self.get_src_path(exp_env)
        dataset = get_dataset_cached(exp_env.dataset)
        with open(path, 'r') as f:
            results = json.load(f)
        assert len(results) == len(dataset), f"LineCheck: Exp {exp_env.get_name()}. {len(results)} != {len(dataset)}"
        res = [self.check_func(output=o, **item) for o, item in zip(results, dataset)]
        with open(self.get_tgt_path(exp_env), 'w') as f:
            json.dump(res, f, indent=2)

        
        
class BinSum(Evaluator):
    def __init__(self, sub_src_file:str, sub_tgt_file:str):
        super().__init__(sub_src_file, sub_tgt_file)

    def evaluate(self, exp_env: ExpEnv):
        path = self.get_src_path(exp_env)
        with open(path, 'r') as f:
            result = json.load(f)
        bins = defaultdict(int)
        for l in result:
            bins[str(l)] += 1
        res_path = self.get_tgt_path(exp_env)
        with open(res_path, 'w') as f:
            json.dump(bins, f, indent=2)




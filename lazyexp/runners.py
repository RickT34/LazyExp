from .exenv import *
from .exper import RUNNER_TYPE, RunnerEnv
from collections import defaultdict
from typing import Callable
from . import envloader
from typing import Any
import subprocess

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

    def runner(self, runner_env: RunnerEnv):
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
        self._runner = runner
        self.model_output_field = model_output_field

    def runner(self, runner_env: RunnerEnv):
        exp_env = runner_env.exp_env

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
        env.dump()
        runner_env_new = RunnerEnv(env, runner_env.environ, env.get_output_path("llm_judge.log"))
        return self._runner(runner_env_new)

class LineCheck(Evaluator):
    def __init__(self, check_func, sub_src_file:str, sub_tgt_file:str):
        self.check_func = check_func
        super().__init__(sub_src_file, sub_tgt_file)
    
    def runner(self, runner_env: RunnerEnv):
        exp_env = runner_env.exp_env
        path = self.get_src_path(exp_env)
        dataset = get_dataset_cached(exp_env.dataset)
        with open(path, 'r') as f:
            results = json.load(f)
        assert len(results) == len(dataset), f"LineCheck: Exp {exp_env.get_name()}. {len(results)} != {len(dataset)}"
        res = [self.check_func(output=o, **item) for o, item in zip(results, dataset)]
        with open(self.get_tgt_path(exp_env), 'w') as f:
            json.dump(res, f, indent=2)
        return 0


class BinSum(Evaluator):
    def __init__(self, sub_src_file:str, sub_tgt_file:str):
        super().__init__(sub_src_file, sub_tgt_file)

    def runner(self, runner_env: RunnerEnv):
        exp_env = runner_env.exp_env
        path = self.get_src_path(exp_env)
        with open(path, 'r') as f:
            result = json.load(f)
        bins = defaultdict(int)
        for l in result:
            bins[str(l)] += 1
        res_path = self.get_tgt_path(exp_env)
        with open(res_path, 'w') as f:
            json.dump(bins, f, indent=2)
        return 0

def cmd_runner(cmd_func: Callable[[ExpEnv], list[str]]) -> RUNNER_TYPE:
    def runner(runner_env: RunnerEnv):
        exp_env = runner_env.exp_env
        environ = runner_env.environ
        log_path = runner_env.log_path
        cmd = cmd_func(exp_env)
        with open(log_path, 'w') as f:
            process = subprocess.Popen(cmd, env=environ, stdout=f, stderr=f)
            process.wait()
            return process.returncode
    return runner

def skip_if_output_exists(runner: RUNNER_TYPE, sub_file:str = "result.json") -> RUNNER_TYPE:
    def new_runner(runner_env: RunnerEnv):
        output_path = runner_env.exp_env.get_output_path(sub_file)
        if os.path.exists(output_path):
            print(f"Output path {output_path} already exists. Skipping.")
            return 0
        else:
            return runner(runner_env)
    return new_runner



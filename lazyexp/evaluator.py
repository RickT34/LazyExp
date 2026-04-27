from .exenv import *
from .exper import run_exps, RUNNER_TYPE
from typing import Callable


class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, exp_env: ExpEnv):
        raise NotImplementedError


class LLMEvaluator(Evaluator):
    def __init__(self, judger: ModelEnv, prompt_template: str, runner: RUNNER_TYPE):
        self.judger = envCopy(judger, ModelEnv)
        self.prompt_template = prompt_template
        self.runner = runner

    def evaluate(self, exp_env: ExpEnv):
        dataset_new = envCopy(exp_env.dataset, DatasetEnv)
        dataset_new.prompt_template = self.prompt_template
        dataset_new.tags["load_hooks"] = [
            ("loader_llm_eval", {"output_path": exp_env.get_output_path()})
        ]
        env = ExpEnv(
            model=self.judger,
            dataset=dataset_new,
            algo=AlgoEnv("loader_llm_eval"),
            label="llmeval",
            output_dir=exp_env.get_output_path("llmeval"),
        )
        self.runner(env)

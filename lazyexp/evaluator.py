from .exenv import *
from .exper import run_exps
import vllmeval

class Evaluator:
    def __init__(self):
        pass
        
    def evaluate(self):
        raise NotImplementedError
    
class LLMEvaluator(Evaluator):
    def __init__(self, exp_env: ExpEnv, judger:ModelEnv, prompt_template:str):
        dataset_new = envCopy(exp_env.dataset, DatasetEnv)
        dataset_new.prompt_template = prompt_template
        dataset_new.tags["load_hooks"] = [("loader_llm_eval", {"output_path": exp_env.get_output_path()})]
        self.env = ExpEnv(
            model=judger,
            dataset=dataset_new,
            algo=AlgoEnv('loader_llm_eval'),
            label="llmeval",
            outputs_dir=exp_env.get_output_path("llmeval")
        )
        self.name = exp_env.get_name()+"_llmeval"
        
    def evaluate(self):
        run_exps([self.env], lambda env: vllmeval.main(env), name=self.name)
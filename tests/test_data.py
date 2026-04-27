from lazyexp.exenv import *
from lazyexp.exper import run_exps
import os
from lazyexp.evaluator import LLMEvaluator
from lazyexp import envloader

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def load_model(modelenv:ModelEnv):
    i = 0
    def model(input:str):
        nonlocal i
        i += 1
        print(f"Model received input: {input}")
        return input.replace("?", "!"*i)
    return model

def load_tokenizer(modelenv:ModelEnv):
    return lambda x: x

def load_dataset(datasetenv:DatasetEnv):
    return [{"input": "What is the capital of France?", "answer": "Paris"},
            {"input": "What is 2+2?", "answer": "4"},
            {"input": "What is the color of the sky?", "answer": "Blue"}]
    
test_model = ModelEnv(
    name="test_model",
    path="tests/test_data.py",
    layers=0,
    filetype="src")
test_dataset = DatasetEnv(
    path="tests/test_data.py",
    filetype="src",
    name="test_dataset",
    prompt_template="Q: {input}\nA: ")

def test_runner(env:ExpEnv):
    if os.path.exists(env.get_output_path()):
        print(f"Skipping {env}.")
        return
    model, tokenizer = envloader.load_model(env.model)
    inputs = envloader.load_inputs(env.dataset)
    outputs = []
    for item in inputs:
        output = model(item)
        outputs.append(output)
    print("Save to: ", env.get_output_path())
    with open(env.get_output_path(), "w") as f:
        json.dump(outputs, f, indent=4)
        
def main_test():
    envs = [ExpEnv(
        model=test_model,
        dataset=test_dataset,
        algo=AlgoEnv("test_algo"),
        label=f"test_label_{i}",
    ) for i in range(10)]
    run_exps(envs, test_runner, name="test_exp")
    evalator = LLMEvaluator( test_model, "Q: {input}\nA: {output}\nJudge: ", test_runner)
    run_exps(envs, evalator.evaluate, name="test_eval")
    
    
if __name__ == "__main__":
    main_test()
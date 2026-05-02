from .exenv import *
from collections import defaultdict
from typing import Callable
from . import envloader
from typing import Any
import subprocess
from pathlib import Path
import time
from .utils import redirect_out_to_file
import os
import sys
import traceback

dataset_cache: dict[str, Any] = {}


def get_dataset_cached(dataset: DatasetEnv):
    if dataset.path not in dataset_cache:
        dataset_cache[dataset.path] = envloader.load_dataset(dataset)
    ds = dataset_cache[dataset.path]
    return ds


class Runner:
    def __init__(
        self, name: str, required_paths: list[Path] = [], output_paths: list[Path] = []
    ):
        self.name = name
        self.required_paths = required_paths
        self.output_paths = output_paths

    def run(self, exp_env: ExpEnv):
        raise NotImplementedError


class Workflow(Runner):
    def __init__(
        self,
        name: str,
        steps: list[Runner],
        skip_success: bool,
        skip_exclude_paths: list[str] = [],
        promised_paths: list[str] = [],
        logs_dir: str | Path = "logs",
    ):
        self.steps = steps
        self.skip_success = skip_success
        self.logs_dir = Path(logs_dir)
        self.skip_exclude_paths = skip_exclude_paths
        super().__init__(name, [Path(p) for p in promised_paths], [])
        self.output_paths = self._check_paths()

    def _check_paths(self):
        current_paths = set()
        for step in self.steps:
            for p in step.required_paths:
                if p not in current_paths and p not in self.required_paths:
                    raise FileNotFoundError(
                        f"Step {step.name} requires path {p} which is not produced by previous steps."
                    )
            for p in step.output_paths:
                if p in current_paths:
                    print(
                        f"Warning: Step {step.name} produces path {p} which is already produced by previous steps."
                    )
                current_paths.add(p)
                p.parent.mkdir(parents=True, exist_ok=True)
        return list(current_paths)

    def run(self, exp_env: ExpEnv):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        logs_dir = exp_env.get_output_path(self.logs_dir) / f"{self.name}_{timestamp}"
        logs_dir.mkdir(parents=True, exist_ok=True)
        for i, step in enumerate(self.steps):
            if self.skip_success:
                if all(
                    exp_env.get_output_path(p).exists()
                    and p.as_posix() not in self.skip_exclude_paths
                    for p in step.output_paths
                ):
                    print(f"Step {step.name} already completed, skipping.")
                    continue
            for p in step.required_paths:
                if not exp_env.get_output_path(p).exists():
                    raise FileNotFoundError(
                        f"Step {step.name} requires path {p} which does not exist."
                    )
            for p in step.output_paths:
                if not p.parent.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
            step_name = f"{i}_{step.name}"
            log_path = logs_dir / f"{step_name}.log"
            with redirect_out_to_file(log_path):
                try:
                    step.run(exp_env)
                except Exception as e:
                    e.add_note(f"Step {step_name} failed.")
                    traceback.print_exc()
                    raise e
            print(f"Step {step_name} completed.")

    def clear_outputs(
        self, exp_envs: list[ExpEnv], base_dir="outputs", dry_dun: bool = True
    ):
        move_envs(
            exp_envs,
            base_dir,
            base_dir + ".bak",
            sub_paths=self.output_paths,
            dry_run=dry_dun,
        )

    def pack_outputs(self, exp_envs:list[ExpEnv], dry_dun: bool = True):
        pack_envs(exp_envs, self.output_paths, output_file=f"{self.name}_outputs.zip", dry_run=dry_dun)

    def info(self):
        print(f"Workflow {self.name}:")
        print(f" Requires: {[str(p) for p in self.required_paths]}")
        for i, step in enumerate(self.steps):
            print(f"  Step {i}: {step.name}")
            print(f"    Requires: {[str(p) for p in step.required_paths]}")
            print(f"    Outputs: {[str(p) for p in step.output_paths]}")
        print(f" Outputs: {[str(p) for p in self.output_paths]}")


class LLMEvalEnv(Runner):
    def __init__(
        self,
        judger: ModelEnv,
        prompt_template: str,
        sub_src_file: str = "result.json",
        sub_tgt_dir: str = "llmeval",
        model_output_field: str = "output",
    ):
        self.sub_tgt_dir = Path(sub_tgt_dir)
        super().__init__(
            "llmeval_mkenv", [Path(sub_src_file)], [self.sub_tgt_dir / "env.json"]
        )
        self.judger = envCopy(judger, ModelEnv)
        self.prompt_template = prompt_template
        self.model_output_field = model_output_field

    def run(self, exp_env):
        path = exp_env.get_output_path(self.required_paths[0])
        dataset_new = envCopy(exp_env.dataset, DatasetEnv)
        dataset_new.prompt_template = self.prompt_template
        dataset_new.tags["load_hooks"] = [
            (
                "loader_llm_eval",
                {
                    "output_path": path.as_posix(),
                    "output_field": self.model_output_field,
                },
            )
        ]
        env = ExpEnv(
            model=self.judger,
            dataset=dataset_new,
            algo=AlgoEnv("loader_llm_eval"),
            label="llmeval",
            output_dir=exp_env.get_output_path(self.sub_tgt_dir).as_posix(),
        )
        env_path = exp_env.get_output_path(self.output_paths[0])
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env.dump(env_path)


class LineCheck(Runner):
    def __init__(self, check_func, sub_src_file: str, sub_tgt_file: str):
        """LineCheck

        Args:
            check_func (Callable[[Any, Any], Any]): (output, sample) -> result
            sub_src_file (str): _description_
            sub_tgt_file (str): _description_
        """
        self.check_func = check_func
        super().__init__("linecheck", [Path(sub_src_file)], [Path(sub_tgt_file)])

    def run(self, exp_env: ExpEnv):
        path = exp_env.get_output_path(self.required_paths[0])
        dataset = get_dataset_cached(exp_env.dataset)
        with open(path, "r") as f:
            results = json.load(f)
        assert len(results) == len(
            dataset
        ), f"LineCheck: Exp {exp_env.get_name()}. {len(results)} != {len(dataset)}"
        res = [self.check_func(o, sample) for o, sample in zip(results, dataset)]
        with open(exp_env.get_output_path(self.output_paths[0]), "w") as f:
            json.dump(res, f, indent=2)


class CmdExec(Runner):
    def __init__(
        self,
        cmd_func: Callable[[ExpEnv], list[str]],
        required_paths: list[Path],
        output_paths: list[Path],
        name: str = "cmd",
    ):
        super().__init__(name, required_paths, output_paths)
        self.cmd_func = cmd_func

    def run(self, exp_env: ExpEnv):
        cmd = self.cmd_func(exp_env)
        process = subprocess.Popen(cmd, env=os.environ)
        process.wait()
        return process.returncode


class EmvDump(Runner):
    def __init__(self, output_path: str = "env.json"):
        super().__init__("envdump", [], [Path(output_path)])

    def run(self, exp_env: envloader.ExpEnv):
        exp_env.dump(exp_env.get_output_path(self.output_paths[0]))


class SummeryTable(Runner):
    def __init__(self, sub_src_file: str, output_file: str = "summery.json"):
        super().__init__("summerytable", [Path(sub_src_file)], [Path(output_file)])

    def run(self, exp_env: ExpEnv):
        path = exp_env.get_output_path(self.required_paths[0])
        data: list = json.load(path.open())
        assert isinstance(data, list)
        summery = defaultdict(list)
        for i, d in enumerate(data):
            if isinstance(d, dict):
                for k, v in d.items():
                    if v is not None:
                        summery[str(k)].append(v)
            else:
                summery[f"n_{str(d)}"].append(i)
        summery_vals = dict()
        for k, v in summery.items():
            if k.startswith("n_"):
                summery_vals[k] = len(v)
            else:
                summery_vals[k] = sum(v) / len(v)
        json.dump(
            summery_vals,
            exp_env.get_output_path(self.output_paths[0]).open("w"),
            indent=2,
        )


def prefab_vllmeval(env_path: str = "env.json", output_path: str = "result.json"):
    return [
        CmdExec(
            cmd_func=lambda env: [
                sys.executable,
                "-m",
                "lazyexp.vllmeval",
                "--env",
                env.get_output_path(env_path).as_posix(),
            ],
            required_paths=[Path(env_path)],
            output_paths=[Path(output_path)],
            name="vllm_runner",
        )
    ]


def prefab_llmeval(
    judger: ModelEnv, prompt_template: str, model_output_field: str = "output"
):
    return [
        LLMEvalEnv(
            judger=judger,
            prompt_template=prompt_template,
            model_output_field=model_output_field,
        ),
        *prefab_vllmeval("llmeval/env.json", "llmeval/result.json"),
    ]

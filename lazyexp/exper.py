import time
from threading import Thread
from pathlib import Path
from .mail import send_default
from .exenv import ExpEnv
import os
import uuid
from .scheduler import Scheduler, Task
from .scheduler_tui import SchedulerUI
from typing import Callable
from multiprocessing import get_context, Process
import dataclasses
import json
from .runners import Runner
from .utils import get_timestamp, redirect_out_to_file


DIR_EXP_HISTORY = Path("exp_history")


def dumpEnvs(envs: list[ExpEnv], name: str, dir: Path = DIR_EXP_HISTORY):
    """
    Save a list of experiment environments to a JSON file.

    Args:
        envs (list[ExpEnv]): List of ExpEnv objects to be saved.
        name (str): Name of the experiment run, used as the filename.
        dir (Path, optional): Directory to save the history. Defaults to DIR_EXP_HISTORY.

    Returns:
        None
    """
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / f"{name}.json"
    # assert not path.exists(), f"exp {name} already exists"
    l = []
    for e in envs:
        l.append(dataclasses.asdict(e))
    return json.dump(l, open(path, "w"), indent=4)


def loadEnvs(name: str, dir: Path = DIR_EXP_HISTORY) -> list[ExpEnv]:
    """
    Load a list of experiment environments from a JSON file.

    Args:
        name (str): Name of the experiment run.
        dir (Path, optional): Directory where the history is saved. Defaults to DIR_EXP_HISTORY.

    Returns:
        list[ExpEnv]: List of loaded ExpEnv objects.
    """
    path = dir / f"{name}.json"
    l = json.load(open(path, "r"))
    envs = []
    for d in l:
        envs.append(ExpEnv(**d))
    return envs




class GPUTask(Task):
    def __init__(self, env: ExpEnv, runner: Runner):
        super().__init__(env.resources_need, env.get_name())
        self.runner = runner
        self.process = None
        self.env = env

    def start(self, resources: list[int]):
        super().start(resources)
        gpu_str = ",".join(map(str, self.allocated))

        def target():
            try:
                # 打印开始信息
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
                id = uuid.uuid4().hex[:4]
                log_path = self.env.get_output_path(f"exp_{get_timestamp()}.log")
                print(f"    Running [{id}]: {log_path} on GPUs {gpu_str}...")
                start_time = time.time()
                with redirect_out_to_file(log_path):
                    self.runner.run(self.env)
                # 计算运行时间
                duration = time.time() - start_time
                msg = f"    Finished [{id}], Duration {duration:.2f} s."
                self.returncode = 0
                print(msg)
            except Exception as e:
                print(f"    !Experiment error: {e}")

        self.process = get_context("fork").Process(target=target)
        self.process.start()

    def check_finish(self):
        assert self.running
        assert self.process is not None
        if not self.process.is_alive():
            return True
        return False

    def close(self):
        super().close()
        del self.process
        self.process = None


def gen_tasks(
    envs: list[ExpEnv],
    runner: Runner,
):
    """
    Run experiments from a list of experiment environments.

    It creates and schedules GPU tasks based on the environments, manages execution,
    saves the experiment configuration, and optionally reports status via UI and email.

    Args:
        envs (list[ExpEnv]): List of environments representing individual experiments.
        runner (Runner): The runner to execute the experiments.
    """
    assert envs, "No envs to run."
    name = runner.name
    dumpEnvs(envs, name, DIR_EXP_HISTORY)
    tasks = []
    for env in envs:
        task = GPUTask(
            runner=runner,
            env=env,
        )
        tasks.append(task)
    return tasks


def run_tasks(tasks: list[Task], ui: bool = True, send_mail: bool = False):
    devices = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
    assert devices, "No CUDA_VISIBLE_DEVICES set."
    scheduler = Scheduler(resources=devices, tasks=tasks)

    if ui:
        try:
            sui = SchedulerUI(scheduler, title="Experiment Scheduler")
            sui.run()
        except Exception as e:
            print(f"Scheduler UI error: {e}, fallback to non-UI mode.")
            scheduler.run()
    else:
        scheduler.run()

    fails = [t.name for t in scheduler.failed_tasks]
    try:
        if fails:
            summery = f"Exp Fails: \n{'\n'.join([str(e) for e in fails])}"
        else:
            summery = "All Experiments Succeeded."
        print(summery)
        if send_mail:
            send_default("ICT-v2", summery)
    except Exception as e:
        print("Failed to send email:", e)

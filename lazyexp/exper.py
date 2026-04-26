import subprocess
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
from multiprocessing import Process, get_context
import dataclasses
import json
import contextlib


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


def get_timestamp():
    """
    Get the current timestamp as a formatted string.

    Returns:
        str: Current time in 'YYYYMMDD_HHMMSS' format.
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


class GPUTask(Task):
    def __init__(self, need: int, name: str, runner: Callable, output_file: str):
        super().__init__(need, name)
        self.runner = runner
        self.output_file = output_file
        self.thread: Thread | None = None

    def start(self, resources: list[int]):
        super().start(resources)
        gpu_str = ",".join(map(str, self.allocated))

        def target():
            try:
                # 打印开始信息
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
                id = uuid.uuid4().hex[:4]
                print(f"    Running [{id}]: {self.output_file}")
                start_time = time.time()

                # 执行命令并捕获输出
                with open(self.output_file, "w") as f:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        self.runner()
                    # 计算运行时间
                    duration = time.time() - start_time
                    msg = f"    Finished [{id}]: Duration {duration:.2f} s."
                    print(msg)
                    f.write(f"\n\n=== {msg} ===\n")
            except Exception as e:
                print(f"    !Experiment error: {e}")

        self.process = get_context("fork").Process(target=target)
        self.process.start()

    def check_finish(self):
        assert self.running
        assert self.process is not None
        return not self.process.is_alive()

    def close(self):
        super().close()
        del self.process
        self.process = None


def run_exps(
    envs: list[ExpEnv],
    runner: Callable[[ExpEnv], None],
    name: str | None = None,
    send_mail: bool = True,
    skip_exist: bool = True,
    ui: bool = True,
):
    """
    Run experiments from a list of experiment environments.

    It creates and schedules GPU tasks based on the environments, manages execution,
    saves the experiment configuration, and optionally reports status via UI and email.

    Args:
        envs (list[ExpEnv]): List of environments representing individual experiments.
        cmd_maker (Callable[[str], list[str]]): A function that takes an environment config path
            and returns a command as a list of strings to be executed.
        name (str | None, optional): The name of the entire experiment run. Defaults to inferred label.
        send_mail (bool, optional): Whether to send a completion notification email. Defaults to True.
        skip_exist (bool, optional): Whether to skip environments whose output already exists. Defaults to True.
        ui (bool, optional): Whether to display a TUI for the scheduler. Defaults to True.
    """
    assert envs, "No envs to run."
    devices = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
    assert devices, "No CUDA_VISIBLE_DEVICES set."
    if name is None:
        labels = {e.label for e in envs}
        if len(labels) == 1:
            name = labels.pop()
    if name is not None:
        dumpEnvs(envs, name, DIR_EXP_HISTORY)
    else:
        print("Warning: Not save envs because of inconsistent labels.")
    tasks = []
    for env in envs:
        if skip_exist and os.path.exists(env.get_output_path()):
            print(f"Skipping {env}.")
            continue
        envpath = env.get_output_path("env.json")
        env.dump(envpath)
        task = GPUTask(
            need=env.model.tags.get("gpus_alloc", 1),
            runner=lambda: runner(env),
            name=env.get_name(),
            output_file=env.get_output_path(f"exp_{get_timestamp()}.log"),
        )
        tasks.append(task)
    scheduler = Scheduler(resources=devices, tasks=tasks)

    if ui:
        try:
            sui = SchedulerUI(scheduler, title=name)
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

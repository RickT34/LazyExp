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
import dataclasses
import json


DIR_EXP_HISTORY = Path("exp_history")

def dumpEnvs(envs: list[ExpEnv], name:str, dir: Path=DIR_EXP_HISTORY):
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / f"{name}.json"
    #assert not path.exists(), f"exp {name} already exists"
    l = []
    for e in envs:
        l.append(dataclasses.asdict(e))
    return json.dump(l, open(path, "w"), indent=4)

def loadEnvs(name:str, dir: Path=DIR_EXP_HISTORY) -> list[ExpEnv]:
    path = dir / f"{name}.json"
    l = json.load(open(path, "r"))
    envs = []
    for d in l:
        envs.append(ExpEnv(**d))
    return envs

def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


class GPUTask(Task):
    def __init__(self, need: int, name: str, cmd: list[str], output_file: str):
        super().__init__(need, name)
        self.cmd = cmd
        self.output_file = output_file
        self.thread: Thread | None = None

    def start(self, resources: list[int]):
        super().start(resources)
        gpu_str = ",".join(map(str, self.allocated))

        def target():
            try:
                # 打印开始信息
                id = uuid.uuid4().hex[:4]
                print(f"    Running [{id}]: {self.output_file}")
                start_time = time.time()

                # 执行命令并捕获输出
                with open(self.output_file, "w") as f:
                    process = subprocess.Popen(
                        self.cmd,
                        stdout=f,
                        stderr=f,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str},
                    )
                    process.wait()  # 等待进程完成

                    # 计算运行时间
                    duration = time.time() - start_time
                    msg = f"    Finished [{id}]: Duration {duration:.2f} s. Code: {process.returncode}"
                    self.returncode = process.returncode
                    print(msg)
                    f.write(f"\n\n=== {msg} ===\n")
                    f.write(f"Experiment command: {' '.join(self.cmd)}\n")
            except Exception as e:
                print(f"    !Experiment error: {self.cmd} : {e}")

        self.thread = Thread(target=target)
        self.thread.start()

    def check_finish(self):
        assert self.running
        assert self.thread is not None
        return not self.thread.is_alive()

    def close(self):
        super().close()
        del self.thread
        self.thread = None


def run_exps(
    envs: list[ExpEnv],
    cmd_maker: Callable[[str], list[str]],
    name: str | None = None,
    send_mail: bool = True,
    skip_exist: bool = True,
    ui: bool = True,
):
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
            cmd=cmd_maker(env.get_output_path("env.json")),
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

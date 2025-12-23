import subprocess
import time
from threading import Thread
from pathlib import Path
from .mail import send_default
from .lazyenv import ExpEnv, dumpEnvs
import os
import uuid


DIR_EXP_HISTORY = Path('exp_history')

def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


class Task:
    def __init__(self, need: int):
        self.need = need
        self.running = False
        self.allocated = []

    def start(self, resources: list[int]):
        assert not self.running
        assert len(resources) >= self.need
        self.running = True
        self.allocated = resources.copy()

    def check_finish(self) -> bool:
        raise NotImplementedError()

    def close(self):
        self.running = False


class GPUTask(Task):
    def __init__(self, need: int, cmd: list[str], output_file: Path):
        super().__init__(need)
        self.cmd = cmd
        self.output_file = output_file
        self.thread: Thread | None = None
        self.returncode: int | None = None

    def start(self, resources: list[int]):
        super().start(resources)
        gpu_str = ",".join(map(str, self.allocated))

        def target():
            try:
                # 输出文件路径处理
                if not self.output_file.parent.exists():
                    self.output_file.parent.mkdir(parents=True)
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


class Scheduler:
    def __init__(self, resources: list[int], tasks: list[Task]):
        self.resources = resources.copy()
        self.tasks = tasks.copy()
        self.running_tasks = []
        self.tasks.sort(key=lambda x: x.need, reverse=True)

    def _check_runnings(self):
        finished = []
        for t in self.running_tasks:
            if t.check_finish():
                t.close()
                finished.append(t)
        for t in finished:
            self.running_tasks.remove(t)
            self.resources.extend(t.allocated)

    def do_schedule(self):
        self._check_runnings()
        while self.resources and self.tasks:
            for t in self.tasks:
                if t.need <= len(self.resources):
                    alloc = self.resources[: t.need]
                    self.resources = self.resources[t.need :]
                    print(f"Scheduler: allocated {alloc}")
                    t.start(alloc)
                    self.running_tasks.append(t)
                    self.tasks.remove(t)
                    break
            else:
                break

    def run(self):
        while self.tasks or self.running_tasks:
            self.do_schedule()
            time.sleep(1)


def run_exps(
    envs: list[ExpEnv],
    cmd_maker,
    name:str|None = None,
    mailsend: bool = True,
    skip_exist: bool = True,
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
        if skip_exist and env.get_output_path().exists():
            print(f"Skipping {env}.")
            continue
        envpath = env.get_output_path("env.json")
        env.dump(envpath)
        task = GPUTask(
            need=env.model.tags.get("gpus_alloc", 1),
            cmd=cmd_maker(env.get_output_path("env.json")),
            output_file=env.get_output_path(f"exp_{get_timestamp()}.log"),
        )
        tasks.append(task)
    scheduler = Scheduler(resources=devices, tasks=tasks)

    scheduler.run()

    fails = [e for e in envs if not e.get_output_path().exists()]
    try:
        summery = f"Exp Done: \n{'\n'.join([str(e) for e in envs])}\n\nFails: \n{'\n'.join([str(e) for e in fails])}"
        print(summery)
        if mailsend:
            send_default("ICT-v2", summery)
    except Exception as e:
        print("Failed to send email:", e)

if __name__ == "__main__":
    # test code
    tasks = [Task(need=2), Task(need=1), Task(need=3)]
    scheduler = Scheduler(resources=[0, 1, 2, 3, 4], tasks=tasks)
    scheduler.run()
import subprocess
import time
from threading import Thread
from pathlib import Path
from .mail import send_default
from .lazyenv import ExpEnv


def run_cmd(command: list[str], output_file: Path):
    """运行单个实验并将输出重定向到文件"""
    try:
        # 输出文件路径处理
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)

        # 打印开始信息
        print(f"    开始实验: {output_file}")
        start_time = time.time()

        # 执行命令并捕获输出
        with open(output_file, "w") as f:
            process = subprocess.Popen(
                command,
                stdout=f,
                stderr=f,
            )
            process.wait()  # 等待进程完成

            # 计算运行时间
            duration = time.time() - start_time
            msg = f"    实验完成: 耗时 {duration:.2f} 秒. 状态码: {process.returncode}"
            f.write(f"\n\n=== {msg} ===\n")
            f.write(f"实验命令: {' '.join(command)}\n")
            return process.returncode
    except Exception as e:
        print(f"实验错误: {command} : {e}")
        return 1

def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
def run_exps(envs: list[ExpEnv], devices:list[int], cmd_maker, mailsend:bool=True):
    running:dict[int, tuple[Thread, ExpEnv]] = {}
    def on_finish(pe):
        pass
    def alloc_device():
        for i in devices:
            if i not in running:
                return i
            elif not running[i][0].is_alive():
                on_finish(running.pop(i))
                return i
        return None
        
    for env in envs:
        if env.get_output_path().exists():
            print(f"Skipping {env} because output file exists.")
            continue
        d = None
        while d is None:
            time.sleep(3)
            d = alloc_device()
        envpath = env.get_output_path("env.json")
        env.dump(envpath)
        cmd = cmd_maker(envpath, d)
        logdir = env.get_output_dir()
        log_file = logdir / f"exp_{get_timestamp()}.log"
        p = Thread(target=run_cmd, args=(cmd, log_file))
        p.start()
        running[d]=(p,env)
    for v in running.values():
        v[0].join()
        on_finish(v)
    try:
        if mailsend:
            send_default("ICT-v2", f"Exp Done: {envs}")
    except:
        print("Failed to send email.")
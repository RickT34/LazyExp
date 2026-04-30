import importlib.util
import sys
import os
from contextlib import contextmanager
import time

_module_count = 0
def call_function_from_file(file_path, function_name, *args, **kwargs):
    # 从文件路径创建模块规范
    global _module_count
    module_name = f"file_module_{_module_count}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"Failed to create spec for {file_path}"
    # 创建模块
    module = importlib.util.module_from_spec(spec)
    # 将模块添加到 sys.modules
    sys.modules[module_name] = module
    # 执行模块中的代码
    spec.loader.exec_module(module)
    _module_count += 1
    # 调用函数
    if not hasattr(module, function_name):
        raise ValueError(f"Function {function_name} not found in {file_path}")
    return getattr(module, function_name)(*args, **kwargs)

def rm_empty_dirs(path):
    if not os.path.isdir(path):
        return
    if not os.listdir(path):
        os.rmdir(path)
        parent = os.path.dirname(path)
        rm_empty_dirs(parent)

def get_timestamp():
    """
    Get the current timestamp as a formatted string.

    Returns:
        str: Current time in 'YYYYMMDD_HHMMSS' format.
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

@contextmanager
def redirect_out_to_file(file_path):
    # 1. 刷新当前的 Python 输出缓冲区，防止原有数据被错误地写入文件
    sys.stdout.flush()

    # 2. 复制并保存原始的 标准输出（FD = 1）
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)

    try:
        # 3. 以写入模式打开目标文件
        with open(file_path, "w") as f:
            # 4. 关键步：将目标文件的文件描述符覆盖到标准输出（FD = 1）上
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)

            # 将执行权交还给 with 代码块
            yield

    finally:
        # 5. 代码块执行完毕，再次刷新缓冲区，确保所有内容已写入文件
        sys.stdout.flush()

        # 6. 将原先保存的标准输出恢复到 FD 1
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)

        # 7. 关闭保存的原始描述符副本，防止资源泄漏
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)

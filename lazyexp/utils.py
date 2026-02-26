import importlib.util
import sys
import os

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
    for root, dirs, files in os.walk(path, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
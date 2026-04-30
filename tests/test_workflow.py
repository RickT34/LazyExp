from lazyexp.runners import *

class TestRunner(Runner):
    def __init__(self, name: str, required_paths: list[Path], output_paths: list[Path]):
        super().__init__(name, required_paths, output_paths)

    def run(self, exp_env: ExpEnv):
        print(f"Running {self.name} with required paths {[str(p) for p in self.required_paths]} and output paths {[str(p) for p in self.output_paths]}")
        strs = ""
        for p in self.required_paths:
            with open(p, "r") as f:
                content = f.read()
                print(f"Content of {p}: {content}")
                strs += content
        for p in self.output_paths:
            with open(p, "w") as f:
                f.write(strs + f"Output from {self.name}\n")
        return 0
    
sub_process_runner = CmdExec(
    cmd_func=lambda env: [
        sys.executable,
        "-c",
        "print('Hello from subprocess!')",
    ],
    required_paths=[],
    output_paths=[],
    name="sub_process_runner",
)

workflow = Workflow("test_workflow", [sub_process_runner, TestRunner("test_runner", [], [Path("output.txt")]), TestRunner("test_runner2", [Path("output.txt")], [Path("output2.txt")])], False)
workflow.info()
workflow.run(ExpEnv(
    model=ModelEnv("test_model", "/path/to/model", 32),
    dataset=DatasetEnv("test_dataset"),
    algo=AlgoEnv("test_algo", {}),
    label="test_label",
    output_dir="./test_output",
))
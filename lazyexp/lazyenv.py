from pathlib import Path
import dataclasses
import os
import json
import itertools
import math
import re


@dataclasses.dataclass
class ModelEnv:
    name: str
    path: str
    layers: int
    tags: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not os.path.exists(self.path):
            print(f"Warning: Model {self.name} not found at {self.path}")

    def get_layers_scattered(self, step: int, count:int = 1):
        i = 0
        while i < self.layers:
            yield list(range(i, min(self.layers, i + count)))
            i += step
        if i - step + count < self.layers:
            yield [self.layers-1]


@dataclasses.dataclass
class DatasetEnv:
    path: str
    range: str = "All"
    tags: dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def get_ds_name(data_json: str):
        ds_name = os.path.basename(data_json)
        ds_name = ds_name[: ds_name.rfind(".")]
        return ds_name

    def __post_init__(self):
        if not os.path.exists(self.path):
            print(f"Warning: Dataset {self.path} not found")
        self.name = self.get_ds_name(self.path)
        if not re.match(r"^\d+[:q]\d+$", self.range) and self.range != "All":
            raise ValueError(f"Dataset range {self.range} invalid")
        if self.range == "1q1":
            self.range = "All"
            
    @staticmethod
    def ds_split(l:int, m:int, n:int):
        k = math.trunc(l/n)
        return k*(m-1), l if m==n else k*m
    
    def read(self):
        return json.load(open(self.path, "r"))
    
    def get_ds_slice(self, l):
        if self.range == "All":
            return slice(0, l)
        elif "q" in self.range:
            m, n = self.range.split("q")
            m = int(m)
            n = int(n)
            start, end = self.ds_split(l, m, n)
            return slice(start, end)
        else:
            m, n = self.range.split(":")
            m = int(m)
            n = int(n)
            return slice(m, n)


@dataclasses.dataclass
class AlgoEnv:
    name: str
    tags: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ExpEnv:
    model: ModelEnv
    dataset: DatasetEnv
    algo: AlgoEnv
    label: str
    outputs_dir: str = "outputs"
    tags: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        for k, c in [
            ("model", ModelEnv),
            ("dataset", DatasetEnv),
            ("algo", AlgoEnv),
        ]:
            v = getattr(self, k)
            if not isinstance(v, c):
                setattr(self, k, c(**v))

        self.filename = f"{self.dataset.range}.json"

    def get_prefile_path(self, prefiles_dir: str):
        prefiledir = Path(prefiles_dir) / self.model.name / self.dataset.name
        prefiledir.mkdir(parents=True, exist_ok=True)
        pre_file = prefiledir / self.filename
        return pre_file

    def get_name(self):
        return f"Exp_{self.model.name}_{self.dataset.name}_{self.label}_{self.dataset.range}"

    def get_output_dir(self):
        outputdir = (
            Path(self.outputs_dir)
            / self.model.name
            / self.dataset.name
            / self.label
            / self.algo.name
        )
        outputdir.mkdir(parents=True, exist_ok=True)
        return outputdir

    def get_output_path(self):
        outputdir = self.get_output_dir()
        output_file = outputdir / self.filename
        return output_file
    
    def to_json(self):
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(json_str):
        d = json.loads(json_str)
        return ExpEnv(**d)
    
    def dump(self, path):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            d = json.load(f)
        return ExpEnv(**d)


ExpHistoryDir = Path("exp_history")

def dumpEnvs(envs: list[ExpEnv], name:str):
    path = ExpHistoryDir / f"{name}.json"
    #assert not path.exists(), f"exp {name} already exists"
    l = []
    for e in envs:
        l.append(dataclasses.asdict(e))
    return json.dump(l, open(path, "w"), indent=4)

def loadEnvs(name:str):
    path = ExpHistoryDir / f"{name}.json"
    l = json.load(open(path, "r"))
    envs = []
    for d in l:
        envs.append(ExpEnv(**d))
    return envs
    

def envCopy(env, cls):
    return cls(**dataclasses.asdict(env))

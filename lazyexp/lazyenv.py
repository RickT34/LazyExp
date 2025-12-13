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
    filetype: str = "hf"

    def check_exists(self):
        return os.path.exists(self.path)

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
    tags: dict = dataclasses.field(default_factory=dict)
    filetype: str = "json"

    @staticmethod
    def get_ds_name(data_json: str):
        ds_name = os.path.basename(data_json)
        ds_name = ds_name[: ds_name.rfind(".")]
        return ds_name
    
    def check_exists(self):
        return os.path.exists(self.path)

    def __post_init__(self):
        self.name = self.get_ds_name(self.path)
            
    @staticmethod
    def ds_split(l:int, m:int, n:int):
        k = math.trunc(l/n)
        return k*(m-1), l if m==n else k*m
    
    def read(self):
        return json.load(open(self.path, "r"))
    
    @staticmethod
    def get_ds_slice(range, l):
        if range == "All":
            return slice(0, l)
        elif "q" in range:
            m, n = range.split("q")
            m = int(m)
            n = int(n)
            start, end = DatasetEnv.ds_split(l, m, n)
            return slice(start, end)
        else:
            m, n = range.split(":")
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
        if not self.dataset.check_exists():
            raise FileNotFoundError(f"Dataset path {self.dataset.path} not found")
        if not self.model.check_exists():
            raise FileNotFoundError(f"Model path {self.model.path} not found")


    def get_name(self):
        return f"Exp_{self.model.name}_{self.dataset.name}_{self.label}"

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

    def get_output_path(self, filename: str = "result.json"):
        outputdir = self.get_output_dir()
        output_file = outputdir / filename
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


def dumpEnvs(envs: list[ExpEnv], name:str, dir: Path):
    path = dir / f"{name}.json"
    #assert not path.exists(), f"exp {name} already exists"
    l = []
    for e in envs:
        l.append(dataclasses.asdict(e))
    return json.dump(l, open(path, "w"), indent=4)

def loadEnvs(name:str, dir: Path) -> list[ExpEnv]:
    path = dir / f"{name}.json"
    l = json.load(open(path, "r"))
    envs = []
    for d in l:
        envs.append(ExpEnv(**d))
    return envs
    

def envCopy(env, cls):
    return cls(**dataclasses.asdict(env))

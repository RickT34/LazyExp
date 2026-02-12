from pathlib import Path
import dataclasses
import os
import json
import itertools
import math
import re
from tqdm import tqdm


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

    def __hash__(self) -> int:
        return hash(repr(self))


@dataclasses.dataclass
class DatasetEnv:
    path: str
    tags: dict = dataclasses.field(default_factory=dict)
    filetype: str = "json"

    @staticmethod
    def get_ds_name(data_json: str):
        ds_name = os.path.basename(data_json)
        return ds_name
    
    @staticmethod
    def get_ds_name_legency(data_json: str):
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
        
    def __hash__(self) -> int:
        return hash(repr(self))


@dataclasses.dataclass
class AlgoEnv:
    name: str
    tags: dict = dataclasses.field(default_factory=dict)
    def __hash__(self) -> int:
        return hash(repr(self))


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
            print(f"Warning: Dataset path {self.dataset.path} not found")
        if not self.model.check_exists():
            print(f"Warning: Model path {self.model.path} not found")
        self.output_dir = (
            Path(self.outputs_dir)
            / self.model.name
            / self.dataset.name
            / self.label
            / self.algo.name
        )


    def get_name(self):
        return f"{self.model.name}_{self.dataset.name}_{self.algo.name}_{self.label}"

    def get_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
    
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

    def __str__(self) -> str:
        return self.get_name()

    def __hash__(self) -> int:
        return hash(repr(self))


    

def envCopy(env, cls):
    return cls(**dataclasses.asdict(env))

def genEnvs(models:list[ModelEnv], datasets:list[DatasetEnv], algos:list[AlgoEnv], label:str, tags:dict={}):
    envs = []
    for model, dataset, algo in itertools.product(models, datasets, algos):
        envs.append(ExpEnv(model=model, dataset=dataset, algo=algo, label=label, tags=tags.copy()))
    return envs

def dl_from_remote(envs: list[ExpEnv], ssh_host:str, remote_base_path:str, filename:str='result.json'):
    reqs = []
    for env in tqdm(envs):
        local_path = env.get_output_path(filename)
        if local_path.exists():
            continue
        reqs.append(local_path.as_posix())
    # pack files on the remote side
    pack_cmd = f'ssh {ssh_host} "tar -czf /tmp/lazyexp_dl.tar.gz -C {remote_base_path} {" ".join(reqs)}"'
    print(f"Packing files on remote side...")
    os.system(pack_cmd)
    # download the packed file
    dl_cmd = f'rsync -ravzP {ssh_host}:/tmp/lazyexp_dl.tar.gz /tmp/lazyexp_dl.tar.gz'
    print(dl_cmd)
    os.system(dl_cmd)
    # unpack files locally
    unpack_cmd = f'tar -xzf /tmp/lazyexp_dl.tar.gz -C {Path.cwd().as_posix()}'
    os.system(unpack_cmd)
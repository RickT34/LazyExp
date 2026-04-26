from pathlib import Path
import dataclasses
import os
import json
import itertools
import math
import re
from tqdm import tqdm
import shutil


@dataclasses.dataclass
class ModelEnv:
    """
    Representation of a model's environment.

    Attributes:
        name (str): Identifier for the model.
        path (str): File path or repository name of the model.
        layers (int): The number of specific layers contained in the model.
        tags (dict): Optional annotations and settings attached to the Model.
        filetype (str): Format of the model file mapping to correct parsers. Default is "hf".
    """

    name: str
    path: str
    layers: int
    tags: dict = dataclasses.field(default_factory=dict)
    filetype: str = "hf"

    def check_exists(self):
        return os.path.exists(self.path)

    def get_layers_scattered(self, step: int, count: int = 1):
        i = 0
        while i < self.layers:
            yield list(range(i, min(self.layers, i + count)))
            i += step
        if i - step + count < self.layers:
            yield [self.layers - 1]

    def __hash__(self) -> int:
        return hash(repr(self))


@dataclasses.dataclass
class DatasetEnv:
    """
    Representation of a dataset's environment.

    Attributes:
        path (str): Local path or HF dataset name.
        tags (dict): Additional annotations/metadata related to the dataset.
        filetype (str): Type of the dataset files (e.g. "json", "hf", "hf_disk"). Default is "json".
        name (str): Extracted or custom name identifier for the dataset.
    """

    path: str
    tags: dict = dataclasses.field(default_factory=dict)
    filetype: str = "json"
    name: str = ""
    prompt_template: str = ""

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
        if self.name == "":
            self.name = self.get_ds_name(self.path)

    @staticmethod
    def ds_split(l: int, m: int, n: int):
        k = math.trunc(l / n)
        return k * (m - 1), l if m == n else k * m

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
    """
    Representation of the specific algorithm running configuration.

    Attributes:
        name (str): Title/Name identifier of the corresponding algorithm configuration.
        tags (dict): Optional dict for custom parameters and algorithm variables.
    """

    name: str
    tags: dict = dataclasses.field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(repr(self))


@dataclasses.dataclass
class ExpEnv:
    """
    Combined encapsulation of a complete experimental environment trial configuration.

    This binds together the Model, Dataset, Algorithm, and a unique label to form an experiment.

    Attributes:
        model (ModelEnv): The Model dependency.
        dataset (DatasetEnv): The Dataset dependency.
        algo (AlgoEnv): The Algorithm definition context.
        label (str): Textual label tagging the experiment, commonly identical across related parallel runs.
        outputs_dir (str): Base output root directory where outcomes and logs will be gathered.
        tags (dict): Optional parameters dict for any broader environment requirements.
        output_dir (str): Final inferred or manually forced exact relative output result folder path.
    """

    model: ModelEnv
    dataset: DatasetEnv
    algo: AlgoEnv
    label: str
    outputs_dir: str = "outputs"
    tags: dict = dataclasses.field(default_factory=dict)
    output_dir: str = ""

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
        if self.output_dir == "":
            self.output_dir = os.path.join(
                self.outputs_dir,
                self.model.name,
                self.dataset.name,
                self.label,
                self.algo.name,
            )

    def get_name(self):
        return f"{self.model.name}_{self.dataset.name}_{self.algo.name}_{self.label}"

    def get_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def get_output_path(self, filename: str = "result.json"):
        return os.path.join(self.get_output_dir(), filename)

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


def genEnvs(
    models: list[ModelEnv],
    datasets: list[DatasetEnv],
    algos: list[AlgoEnv],
    label: str,
    tags: dict = {},
):
    """
    Generate parallel combinations of Experiment Environments (`ExpEnv`) using Cartesian multiplication.

    Args:
        models (list[ModelEnv]): A list of all models to run.
        datasets (list[DatasetEnv]): A list of datasets to run against.
        algos (list[AlgoEnv]): A list of testing algorithm settings.
        label (str): Generic label parameter identical for all these permutations.
        tags (dict, optional): Basic parameter settings transferred equally to all permutations.

    Returns:
        list[ExpEnv]: Formed distinct array containing every parallel combinations configuration.
    """
    envs = []
    for model, dataset, algo in itertools.product(models, datasets, algos):
        envs.append(
            ExpEnv(
                model=model, dataset=dataset, algo=algo, label=label, tags=tags.copy()
            )
        )
    return envs


def dl_from_remote(
    envs: list[ExpEnv],
    ssh_host: str,
    remote_base_path: str,
    filename: str = "result.json",
):
    """
    Downloads and updates local experiment results from a mapped remote server workspace.

    Args:
        envs (list[ExpEnv]): Environments specifying the results needed.
        ssh_host (str): String identifier for SSH host.
        remote_base_path (str): Relative root path corresponding identically on the remote side.
        filename (str, optional): Remote target JSON log file. Defaults to 'result.json'.
    """
    reqs = []
    for env in tqdm(envs):
        local_path = env.get_output_path(filename)
        if os.path.exists(local_path):
            continue
        reqs.append(local_path)
    # pack files on the remote side
    pack_cmd = f'ssh {ssh_host} "tar -czf /tmp/lazyexp_dl.tar.gz -C {remote_base_path} {" ".join(reqs)}"'
    print(f"Packing files on remote side...")
    os.system(pack_cmd)
    # download the packed file
    dl_cmd = f"rsync -ravzP {ssh_host}:/tmp/lazyexp_dl.tar.gz /tmp/lazyexp_dl.tar.gz"
    print(dl_cmd)
    os.system(dl_cmd)
    # unpack files locally
    unpack_cmd = f"tar -xzf /tmp/lazyexp_dl.tar.gz -C {Path.cwd().as_posix()}"
    os.system(unpack_cmd)


def envMove(src: ExpEnv, dst: ExpEnv):
    """
    Move the output dir from src env to dst env
    """
    src_output_dir = src.get_output_dir()
    dst_output_dir = dst.get_output_dir()
    os.rename(src_output_dir, dst_output_dir)

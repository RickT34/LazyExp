import dataclasses
import os
import json
import itertools
import math
import shutil
from pathlib import Path


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
    thinking: bool | None = None
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
    outputs_basedir: str = "outputs"
    tags: dict = dataclasses.field(default_factory=dict)
    output_dir: str = ""
    resources_need: int = 1

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
                self.outputs_basedir,
                self.model.name,
                self.dataset.name,
                self.label,
                self.algo.name,
            )

    def get_name(self):
        return f"{self.model.name}_{self.dataset.name}_{self.algo.name}_{self.label}"

    def get_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        return Path(self.output_dir)

    def get_output_path(self, filename: str | Path = "result.json"):
        return self.get_output_dir() / filename

    def to_json(self):
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(json_str):
        d = json.loads(json_str)
        return ExpEnv(**d)

    def dump(self, path: str | Path | None = None):
        if path is None:
            path = self.get_output_path("env.json")
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)
        return path

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


import zipfile


def pack_envs(
    envs: list[ExpEnv],
    sub_paths: list[Path] = [],
    output_file: str = "outputs.zip",
    dry_run: bool = False,
):
    if sub_paths:
        paths = [e.get_output_path(p) for e in envs for p in sub_paths]
    else:
        paths = [e.get_output_dir() for e in envs]
    with zipfile.ZipFile(output_file, "w") as zipf:
        for path in paths:
            if not path.exists(): continue
            if path.is_dir():
                for root, dirs, files in os.walk(path):
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        if dry_run:
                            print(f"Packing {file_path} -> {output_file}")
                        else:
                            zipf.write(file_path, file_path)
            else:
                if dry_run:
                    print(f"Packing {path} -> {output_file}")
                else:
                    zipf.write(path, path)
    return output_file


def move_envs(
    envs: list[ExpEnv],
    base_dir: str,
    target_dir: str,
    sub_paths: list[Path] = [],
    dry_run: bool = False,
):
    for env in envs:
        files = []
        if sub_paths:
            for f in sub_paths:
                path = env.get_output_path(f)
                if path.exists():
                    files.append(path)
        else:
            path = env.get_output_dir()
            if path.exists():
                files.append(path)
        for src in files:
            dst = os.path.join(target_dir, os.path.relpath(src, base_dir))
            if dry_run:
                print(f"{src} -> {dst}")
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)

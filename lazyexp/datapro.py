from lazyexp.exenv import ExpEnv, DatasetEnv
import random
from lazyexp import envloader, exenv, exper
import json
from datasets import Dataset, concatenate_datasets
import numpy as np
from collections import defaultdict
import pandas as pd
import os
from tqdm import tqdm
from typing import Callable
from matplotlib import pyplot as plt
from matplotlib import axes
import numpy as np
import enum

dataset_cache: dict[str, Dataset] = {}


def get_dataset_cached(dataset: DatasetEnv):
    if dataset.path not in dataset_cache:
        dataset_cache[dataset.path] = envloader.load_dataset(dataset)
    ds = dataset_cache[dataset.path]
    return ds


def process_exps(envs: list[ExpEnv], process_fn: Callable[[ExpEnv], dict]):
    results = []
    for env in tqdm(envs):
        results.append(process_fn(env))
    try:
        df = pd.DataFrame(results)
        return df
    except Exception as e:
        print("Could not create DataFrame, return raw results:", e)
        return results


class ExpAxis(enum.Enum):
    ModelAxis = (0,)
    DatasetAxis = (1,)
    AlgoAxis = (2,)
    LabelAxis = (3,)

    def get_attr_name(self):
        return {
            ExpAxis.ModelAxis: "model",
            ExpAxis.DatasetAxis: "dataset",
            ExpAxis.AlgoAxis: "algo",
            ExpAxis.LabelAxis: "label",
        }[self]


def envs_decompose(envs: list[ExpEnv]):
    models = set()
    datasets = set()
    algos = set()
    labels = set()
    for env in envs:
        models.add(env.model)
        datasets.add(env.dataset)
        algos.add(env.algo)
        labels.add(env.label)
    if len(models) * len(datasets) * len(algos) * len(labels) != len(envs):
        print("Warning: envs are not fully combinatorial.")
    return list(models), list(datasets), list(algos), list(labels)


DEFAULT_PLOT_ARGS = {"linewidth": 2, "markersize": 5, "marker": "o"}

PREDEF_COLORS = ["#F47F72", "#89BF99", "#7FB2D5", "#F7B76D", "#857CDB"]


def get_random_color():
    if PREDEF_COLORS:
        return PREDEF_COLORS.pop(0)
    LB, UB = 0x7F, 0xE4
    r = random.randint(LB, UB)
    g = random.randint(LB, UB)
    b = random.randint(LB, UB)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def explot(
    envs: list[ExpEnv],
    axises: tuple[ExpAxis, ExpAxis, ExpAxis],
    process_fn: Callable,
    xlabel: str = "",
    translator: Callable[[str], str] | None = None,
    plot_args: dict = {},
    ax_hook: Callable[[axes.Axes, list[ExpEnv]], None] | None = None,
):
    models, datasets, algos, labels = envs_decompose(envs)
    axis_map = {
        ExpAxis.ModelAxis: models,
        ExpAxis.DatasetAxis: datasets,
        ExpAxis.AlgoAxis: algos,
        ExpAxis.LabelAxis: labels,
    }
    XX, YY, ZZ = tuple(axis_map[ax] for ax in axises)
    fig, ax = plt.subplots(
        len(YY), len(XX), figsize=(5 * len(XX), 4 * len(YY)), squeeze=False
    )
    colors_mem = {}

    def autoplot(ax: axes.Axes, data, plot_args: dict = {}):
        for k, v in DEFAULT_PLOT_ARGS.items():
            plot_args.setdefault(k, v)
        if isinstance(data, tuple) and len(data) == 2:
            x, y = data
        else:
            x = np.arange(len(data))
            y = data

        if isinstance(x[0], str):
            ax.set_xticklabels(x, rotation=45, ha="right")
            x = np.arange(len(x))
        if len(x) < 50:
            ax.set_xticks(x)
        if isinstance(y, dict):
            for label, yv in y.items():
                pa = plot_args.copy()
                x = np.arange(len(yv))
                if "color" not in pa:
                    pa["color"] = colors_mem.get(label, get_random_color())
                    colors_mem[label] = pa["color"]
                ax.plot(x, yv, label=label, **pa)
        else:
            ax.plot(x, y, **plot_args)
        ax.grid(True)

    if translator is None:
        translator = lambda x: x
    for i, y in enumerate(YY):
        for j, x in enumerate(XX):
            sub_envs = list(
                filter(
                    lambda e: (getattr(e, axises[0].get_attr_name()) == x)
                    and (getattr(e, axises[1].get_attr_name()) == y),
                    envs,
                )
            )
            data = process_fn(sub_envs)
            autoplot(ax[i][j], data, plot_args)
            if j == 0:
                ax[i][j].set_ylabel(translator(y.name), fontsize=14)
            if i == 0:
                ax[i][j].set_title(translator(x.name), fontsize=14)
            elif i == len(YY) - 1:
                ax[i][j].set_xlabel(translator(xlabel), fontsize=12)
            if ax_hook is not None:
                ax_hook(ax[i][j], sub_envs)
    handles, labels = ax[-1][-1].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            ncols=len(labels),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            frameon=False,
            fontsize=12,
        )
        plt.tight_layout(pad=1.0)
        fig.subplots_adjust(top=0.9)
    else:
        plt.tight_layout()
    return fig

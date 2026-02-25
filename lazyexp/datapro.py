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
import itertools
import colorsys

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
    attrs = ("model", "dataset", "algo", "label")
    comps = {attr: [] for attr in attrs}
    for env in envs:
        for attr in attrs:
            val = getattr(env, attr)
            if val not in comps[attr]:
                comps[attr].append(val)
    if np.prod([len(comps[attr]) for attr in attrs]) != len(envs):
        print("Warning: envs are not fully combinatorial.")
    return tuple(comps[attr] for attr in attrs)


DEFAULT_PLOT_ARGS = {"linewidth": 2, "markersize": 5, "marker": "o"}


_color_state = random.random()

def get_random_color():
    """
    生成一个科研风格的随机颜色（Hex格式）。
    每次调用返回一个与之前差异足够大的颜色。
    """
    global _color_state
    # 黄金分割比，用于在色相环上均匀分布
    golden_ratio = (5**0.5 - 1) / 2
    
    # 更新全局色相状态 (利用列表的可变性模拟静态变量)
    _color_state = (_color_state + golden_ratio) % 1
    
    # 参数调整 (HSV空间):
    # H (Hue): 自动计算
    # S (Saturation): 0.6 (适中，不刺眼)
    # V (Value): 0.95 (明亮，适合白底论文)
    h = _color_state
    s = 0.6  
    v = 0.95 
    
    # 转为RGB并生成Hex代码
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def _decompose_envs_by_axis(
    envs: list[ExpEnv], axises: tuple[ExpAxis, ...]
):
    models, datasets, algos, labels = envs_decompose(envs)
    axis_map = {
        ExpAxis.ModelAxis: models,
        ExpAxis.DatasetAxis: datasets,
        ExpAxis.AlgoAxis: algos,
        ExpAxis.LabelAxis: labels,
    }
    axiss = tuple(axis_map[ax] for ax in axises)
    assert len(axiss) >= 2, "At least two axises are required."
    assert len(axiss) <= 3, "At most three axises are supported."
    XX, YY = axiss[0], axiss[1]
    ZZ = axiss[2] if len(axiss) > 2 else [None]
    envs_splited = defaultdict(list)
    for env in envs:
        key = (
            getattr(env, axises[0].get_attr_name()),
            getattr(env, axises[1].get_attr_name()),
            getattr(env, axises[2].get_attr_name()) if len(axiss) == 3 else None,
        )
        envs_splited[key].append(env)
    return XX, YY, ZZ, envs_splited
    


def explot(
    envs: list[ExpEnv],
    axises: tuple[ExpAxis, ...],
    process_fn: Callable,
    xlabel: str = "",
    translator: Callable[[str], str] | None = None,
    plot_args: dict = {},
    ax_hook: Callable[[axes.Axes, list[ExpEnv]], None] | None = None,
    colors: list[str] = [],
):
    XX, YY, ZZ, envs_splited = _decompose_envs_by_axis(envs, axises)
    fig, ax = plt.subplots(
        len(YY), len(XX), figsize=(5 * len(XX), 4 * len(YY)), squeeze=False
    )
    colors_mem = {}
    colors = colors.copy()
    
    def get_color(label: str) -> str:
        if label in colors_mem:
            return colors_mem[label]
        if colors:
            color = colors.pop(0)
        else:
            color = get_random_color()
        colors_mem[label] = color
        return color

    def autoplot(ax: axes.Axes, data, plot_args: dict = {}):
        for k, v in DEFAULT_PLOT_ARGS.items():
            plot_args.setdefault(k, v)
        if isinstance(data, tuple) and len(data) == 2:
            x, y = data
        else:
            x = np.arange(len(data))
            y = data

        if isinstance(x[0], str):
            x2 = np.arange(len(x))
            ax.set_xticks(x2)
            ax.set_xticklabels(x, rotation=45, ha="right")
            x = x2
        elif len(x) < 50:
            ax.set_xticks(x)
        if isinstance(y, dict):
            for label, yv in y.items():
                pa = plot_args.copy()
                x = np.arange(len(yv))
                if "color" not in pa:
                    pa["color"] = get_color(label)
                    colors_mem[label] = pa["color"]
                ax.plot(x, yv, label=label, **pa)
        else:
            ax.plot(x, y, **plot_args)
        ax.grid(True)

    if translator is None:
        translator = lambda x: str(x)
    trans_wrapper = lambda x: translator(x.name if hasattr(x, "name") else str(x))
    for (i, y), (j, x), (k, z) in tqdm(itertools.product(enumerate(YY), enumerate(XX), enumerate(ZZ)), total=len(YY)*len(XX)*len(ZZ)):
        sub_envs = envs_splited[x, y, z]
        if len(sub_envs) == 0:
            print(f"Warning: no envs for subplot ({x}, {y}, {z})")
        else:
            data = process_fn(sub_envs)
            autoplot(ax[i][j], data, plot_args)
        if k == 0:
            if j == 0:
                ax[i][j].set_ylabel(trans_wrapper(y), fontsize=14)
            if i == 0:
                ax[i][j].set_title(trans_wrapper(x), fontsize=14)
            if i == len(YY) - 1:
                ax[i][j].set_xlabel(trans_wrapper(xlabel), fontsize=12)
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

def extable(
    envs: list[ExpEnv],
    axises: tuple[ExpAxis, ...],
    process_fn: Callable,
    translator: Callable[[str], str] | None = None,
):
    XX, YY, ZZ, envs_splited = _decompose_envs_by_axis(envs, axises)
    table_data = {}
    if translator is None:
        translator = lambda x: str(x)
    trans_wrapper = lambda x: translator(x.name if hasattr(x, "name") else str(x))
    for x, y, z in tqdm(itertools.product(XX, YY, ZZ), total=len(XX)*len(YY)*len(ZZ)):
        sub_envs = envs_splited[x, y, z]
        if len(sub_envs) == 0:
            print(f"Warning: no envs for table cell ({x}, {y}, {z})")
            continue
        data = process_fn(sub_envs)
        key_x = trans_wrapper(x)
        key_y = trans_wrapper(y)
        if z is None:
            table_data[key_x, key_y] = data
        else:
            key_z = trans_wrapper(z)
            table_data[key_x, key_y, key_z] = data
    df = pd.DataFrame.from_dict(table_data, orient="index")
    return df

def exshow(
    envs: list[ExpEnv],
    axises: tuple[ExpAxis, ...],
    process_fn: Callable,
):
    XX, YY, ZZ, envs_splited = _decompose_envs_by_axis(envs, axises)
    trans_wrapper = lambda x: x.name if hasattr(x, "name") else str(x)
    selected = []
    for i in (XX, YY, ZZ):
        for x in enumerate(i):
            print(f"{i}: {trans_wrapper(x)}")
        selected.append(i[int(input(f"Select: "))])
    env = envs_splited[selected[0], selected[1], selected[2]]
    for case in process_fn(env):
        for k, v in case.items():
            print(f"========== {k} ==========")
            print(v)
        input("Press Enter to continue...")
    
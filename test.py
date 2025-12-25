from lazyexp import datapro, exenv
import numpy as np

MODELS = [
    exenv.ModelEnv("gpt2-small", path="gpt2", layers=12),
    exenv.ModelEnv("gpt2-medium", path="gpt2-medium", layers=24),
]
DATASETS = [
    exenv.DatasetEnv("data/ds1.json"),
    exenv.DatasetEnv("data/ds2.json"),
]
ALGOS = [
    exenv.AlgoEnv("algo1", tags={"lr": 1e-4}),
    exenv.AlgoEnv("algo2", tags={"lr": 5e-5})
]

envs = exenv.genEnvs(
    models=MODELS,
    datasets=DATASETS,
    algos=ALGOS,
    label="test_experiment",
) + exenv.genEnvs(
    models=MODELS,
    datasets=DATASETS,
    algos=ALGOS,
    label="test_experiment_v2",
)

def process_fn(envs: list[exenv.ExpEnv]):
    return {e.label: np.random.random(10) for e in envs}

fig = datapro.explot(
    envs,
    process_fn=process_fn,
    axises=(datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis, datapro.ExpAxis.LabelAxis),
    xlabel="Layer",
)
fig.savefig("experiment_results.png")
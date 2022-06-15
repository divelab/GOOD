# GOOD
GOOD: A Graph Out-of-Distribution Benchmark

[license-url]: https://github.com/divelab/GOOD/blob/main/LICENSE
[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg

[![Documentation Status](https://readthedocs.org/projects/good/badge/?version=latest)](https://good.readthedocs.io/en/latest/?badge=latest)
![Last Commit](https://img.shields.io/github/last-commit/divelab/DIG)
[![License][license-image]][license-url]
[![codecov](https://codecov.io/gh/divelab/GOOD/branch/main/graph/badge.svg?token=W41HSP0XCY)](https://codecov.io/gh/divelab/GOOD)
[![CircleCI](https://circleci.com/gh/divelab/GOOD/tree/main.svg?style=svg)](https://circleci.com/gh/divelab/GOOD/tree/main)
[![GOOD stars](https://img.shields.io/github/stars/divelab/GOOD?style=social)](https://github.com/divelab/GOOD)

[Documentation](https://good.readthedocs.io)
> We are actively building the document.

* [Overview](#overview)
* [Why GOOD?](#why-good-)
* [Installation](#installation)
* [Quick start](#quick-start)
  * [Module usage (recommended: use only GOOD datasets)](#module-usage)
  * [Project usage (recommended: OOD algorithm researches & developments)](#project-usage)

## Overview

Out-of-distribution (OOD) learning deals with scenarios in which training and test data follow different distributions. 
Although general OOD problems have been intensively studied in machine learning, graph OOD is only an emerging area of research. 
Currently, there lacks a systematic benchmark tailored to graph OOD method evaluation. 
This project is for Graph Out-of-distribution development, known as GOOD.
We explicitly make distinctions between covariate and concept shifts and design data splits that accurately reflect different shifts. 
We consider both graph and node prediction tasks as there are key differences when designing shifts. 
Currently, GOOD contains 8 datasets with 14 domain selections. When combined with covariate, concept, and no shifts, we obtain 42 different splits. 
We provide performance results on 7 commonly used baseline methods with 10 random runs. 
This results in 294 dataset-model combinations in total. Our results show significant performance gaps between in-distribution and OOD settings. 
We hope our results also shed light on different performance trends between covariate and concept shifts by different methods. 
This GOOD benchmark is a growing project and expects to expand in both quantity and variety of resources as the area develops.
Any contribution is welcomed!

## Why GOOD?

Whether you are an experienced researcher for graph out-of-distribution problem or a first-time learner of graph deep learning, 
here are several reasons for you to use GOOD as your Graph OOD research, study, and development toolkit.

* **Easy-to-use APIs:** GOOD provides simple apis for loading OOD algorithms, graph neural networks, and datasets, so that you can take only several lines of code to start.
* **Flexibility:** Full OOD split generalization code is provided for extensions and any new graph OOD dataset contributions.
OOD algorithm base class can be easily overwrite to create new OOD methods.
* **Easy-to-extend architecture:** In addition to play as a package, GOOD is also an integrated and well-organized project ready to be further developed.
All algorithms, models, and datasets can be easily registered by `register` and automatically embedded into the designed pipeline like a breeze!
The only thing the user need to do is writing your own OOD algorithm class, your own model class, or your new dataset class.
Then you can compare your results with the leaderboard.
* **Easy comparisons with the leaderboard:** We provide insightful comparisons from multiple perspectives. Any researches and studies can use
our leaderboard results for comparison. Note that, this is a growing project, so we will include new OOD algorithms gradually.
Besides, if you hope to include your algorithms in the leaderboard, please contact us or contribute to this project. A big welcome!
* **Reproducibility:** 
  * OOD Datasets: GOOD provides full OOD split generalization code for reproduction, and generation for new datasets.
  * Leaderboard results: One random seed round results are provided, and loaded models pass the test result reproduction.


## Installation 

### Conda dependencies

GOOD depends on [PyTorch (>=1.6.0)](https://pytorch.org/get-started/previous-versions/), [PyG (>=2.0)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), and
[RDKit (>=2020.09.5)](https://www.rdkit.org/docs/Install.html). For more details: [conda environment](https://github.com/divelab/GOOD/blob/main/environment.yml)

> Note that we currently test on PyTorch (==1.10.1), PyG (==2.0.3), RDKit (==2020.09.5); thus we strongly encourage to install these versions.
>
> Attention! Due to a known issue, please install PyG through Pip to avoid incompatibility.

### Pip (Beta)

#### Only use modules independently (pending)

```shell
pip install graph-ood
```

#### Take the advantages of whole project (recommended)

```shell
git clone https://github.com/divelab/GOOD.git && cd GOOD
pip install -e .
```

## Quick Start

### Module usage

#### GOOD datasets
There are two ways to import 8 GOOD datasets with 14 domain selections and totally 42 splits:
```python
# Directly import
from GOOD.data.good_datasets.good_hiv import GOODHIV
hiv_datasets = GOODHIV.load(dataset_root, domain='scaffold', shift='covariate', generate=False)
# Or using register
from GOOD import register as good_reg
hiv_datasets = good_reg.datasets['GOODHIV'].load(dataset_root, domain='scaffold', shift='covariate', generate=False)
cmnist_datasets = good_reg.datasets['GOODCMNIST'].load(dataset_root, domain='color', shift='concept', generate=False)
```

#### GOOD GNNs
The best and fair way to compare algorithms with the leaderboard is to use the same and similar graph encoder structure;
therefore, we provide GOOD GNN apis to support. Here, we use an objectified dictionary `config` to pass parameters. More
details about the config: [Document of config (pending)]()

*To use exact GNN*
```python
from GOOD.networks.models.GCNs import GCN
model = GCN(config)
# Or
from GOOD import register as good_reg
model = good_reg.models['GCN'](config)
```
*To only use parts of GNN*
```python
from GOOD.networks.models.GINvirtualnode import GINEncoder
encoder = GINEncoder(config)
```

#### GOOD algorithms
Try to apply OOD algorithms to your own models?
```python
from GOOD.ood_algorithms.algorithms.VREx import VREx
ood_algorithm = VREx(config)
# Then you can provide it to your model for necessary ood parameters, 
# and use its hook-like function to process your input, output, and loss.
```

### Project usage

It is a good beginning to directly make it work. Here, we provide command line script `goodtg` (GOOD to go) to access the main function located at `GOOD.kernel.pipeline:main`.
Choosing a config file in `configs/GOOD_configs`, we can start a task:

```shell
goodtg --config_path GOOD_configs/GOODCMNIST/color/concept/DANN.yaml
```

Specifically, the task is clearly divided into three parts:
1. **Config**
```python
from GOOD import config_summoner
from GOOD.utils.args import args_parser
from GOOD.utils.logger import load_logger
args = args_parser()
config = config_summoner(args)
load_logger(config)
```
2. **Loader**
```python
from GOOD.kernel.pipeline import initialize_model_dataset
from GOOD.ood_algorithms.ood_manager import load_ood_alg
model, loader = initialize_model_dataset(config)
ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
```
*Or concretely,*
```python
from GOOD.data import load_dataset, create_dataloader
from GOOD.networks.model_manager import load_model
from GOOD.ood_algorithms.ood_manager import load_ood_alg
dataset = load_dataset(config.dataset.dataset_name, config)
loader = create_dataloader(dataset, config)
model = load_model(config.model.model_name, config)
ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
```
3. **Train/test pipeline**
```python
from GOOD.kernel.pipeline import load_task
load_task(config.task, model, loader, ood_algorithm, config)
```
*Or concretely,*
```python
# Train
from GOOD.kernel.train import train
train(model, loader, ood_algorithm, config)
# Test
from GOOD.kernel.evaluation import evaluate
test_stat = evaluate(model, loader, ood_algorithm, 'test', config)
```


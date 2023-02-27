r"""A module that is consist of a pipeline loader and model configuration function.
"""

from typing import Dict
from typing import Union

import torch.nn
from munch import Munch
from torch.utils.data import DataLoader

from GOOD.kernel.pipelines.basic_pipeline import Pipeline
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.args import CommonArgs
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.register import register


def load_pipeline(name: str,
                  task: str,
                  model: torch.nn.Module,
                  loader: Union[DataLoader, Dict[str, DataLoader]],
                  ood_algorithm: BaseOODAlg,
                  config: Union[CommonArgs, Munch]
                  ) -> Pipeline:
    r"""
    A pipeline loader.
    Args:
        name (str): Name of the chosen pipeline
        config (Union[CommonArgs, Munch]): Please refer to specific GNNs for required configs and formats.

    Returns:
        A instantiated pipeline.

    """
    try:
        reset_random_seed(config)
        pipeline = register.pipelines[name](task, model, loader, ood_algorithm, config)
    except KeyError as e:
        print(f'#E#Pipeline {name} does not exist.')
        raise e
    return pipeline

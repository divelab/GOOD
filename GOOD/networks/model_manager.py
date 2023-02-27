r"""A module that is consist of a GNN model loader and model configuration function.
"""

import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed


def load_model(name: str, config: Union[CommonArgs, Munch]) -> torch.nn.Module:
    r"""
    A model loader.
    Args:
        name (str): Name of the chosen GNN.
        config (Union[CommonArgs, Munch]): Please refer to specific GNNs for required configs and formats.

    Returns:
        A instantiated GNN model.

    """
    try:
        reset_random_seed(config)
        model = register.models[name](config)
    except KeyError as e:
        print(f'#E#Model {name} does not exist.')
        raise e
    return model




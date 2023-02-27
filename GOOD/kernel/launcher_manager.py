r"""A module that is consist of a launcher loader and model configuration function.
"""

from typing import Dict
from typing import Union

import torch.nn
from munch import Munch
from torch.utils.data import DataLoader

from GOOD.kernel.launchers.basic_launcher import Launcher
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.args import CommonArgs
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.register import register


def load_launcher(name: str) -> Launcher:
    r"""
    A launcher loader.
    Args:
        name (str): Name of the chosen launcher

    Returns:
        A instantiated launcher.

    """
    try:
        launcher = register.launchers[name]()
    except KeyError as e:
        print(f'#E#Launcher {name} does not exist.')
        raise e
    return launcher

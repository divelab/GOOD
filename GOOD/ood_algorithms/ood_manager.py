"""A module that is consist of an OOD algorithm loader.
"""

from GOOD import register
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.config_reader import Union, CommonArgs, Munch


def load_ood_alg(name, config: Union[CommonArgs, Munch]):
    r"""
    OOD algorithm loader.
    Args:
        name: Name of the chosen OOD algorithm.
        config: please refer to specific algorithms for required configs.

    Returns:
        An OOD algorithm object.

    """
    try:
        ood_algorithm: BaseOODAlg = register.ood_algs[name](config)
    except KeyError as e:
        print(f'#E#OOD algorithm of given name does not exist.')
        raise e
    return ood_algorithm

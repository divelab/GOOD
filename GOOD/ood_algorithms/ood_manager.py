"""A module that is consist of an OOD algorithm loader.
"""

from GOOD import register
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.config_reader import Union, CommonArgs, Munch


def load_ood_alg(name, config: Union[CommonArgs, Munch]):
    try:
        ood_algorithm: BaseOODAlg = register.ood_algs[name](config)
    except KeyError as e:
        print(f'#E#OOD algorithm of given name does not exist.')
        raise e
    return ood_algorithm

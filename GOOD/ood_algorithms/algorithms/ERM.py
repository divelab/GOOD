"""
Implementation of the baseline ERM
"""
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


@register.ood_alg_register
class ERM(BaseOODAlg):
    r"""
    Implementation of the baseline ERM

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(ERM, self).__init__(config)

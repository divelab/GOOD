"""
Implementation of the VREx algorithm from `"Out-of-Distribution Generalization via Risk Extrapolation (REx)"
<http://proceedings.mlr.press/v139/krueger21a.html>`_ paper
"""
import torch
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.train import at_stage
from GOOD.utils.initial import reset_random_seed

@register.ood_alg_register
class EERM(BaseOODAlg):
    r"""
    Implementation of the VREx algorithm from `"Out-of-Distribution Generalization via Risk Extrapolation (REx)"
    <http://proceedings.mlr.press/v139/krueger21a.html>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(EERM, self).__init__(config)

    def stage_control(self, config: Union[CommonArgs, Munch]):
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1
            config.train_helper.optimizer = torch.optim.Adam(config.train_helper.model.gnn.parameters(), lr=config.train.lr,
                                                             weight_decay=config.train.weight_decay)
            config.train_helper.scheduler = torch.optim.lr_scheduler.MultiStepLR(config.train_helper.optimizer,
                                                                                 milestones=config.train.mile_stones,
                                                                                 gamma=0.1)

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor, config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func: Accuracy}
                                   })


        Returns (Tensor):
            cross entropy loss

        """
        assert config.model.model_level == 'node'
        return raw_pred

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on VREx algorithm

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)

        .. code-block:: python

            config = munchify({device: torch.device('cuda'),
                                   dataset: {num_envs: int(10)},
                                   ood: {ood_param: float(0.1)}
                                   })


        Returns (Tensor):
            loss based on VREx algorithm

        """
        Var, Mean = loss
        loss = Mean + config.ood.ood_param * Var
        self.mean_loss = Mean
        self.spec_loss = Var
        return loss

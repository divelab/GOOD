"""
"""
import torch
from torch.autograd import grad
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
from typing import Tuple
from torch.nn.functional import cross_entropy
from copy import deepcopy
from GOOD.utils.train import at_stage
from GOOD.utils.initial import reset_random_seed
from GOOD.networks.model_manager import load_model


def compute_covariance(input_data: Tensor) -> Tensor:
    r"""
    Compute Covariance matrix of the input data

    Args:
        input_data (Tensor): feature of the input data
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`)

    .. code-block:: python

        config = munchify({device: torch.device('cuda')})

    Returns (Tensor):
        covariance value of the input features

    """
    n = input_data.shape[0]  # batch_size

    sum_feature = torch.sum(input_data, dim=0, keepdim=True)
    average_matrix = torch.div(torch.mm(sum_feature.t(), sum_feature), n)
    cross_matrix = torch.mm(input_data.t(), input_data)
    cov_matrix = (cross_matrix - average_matrix) / (n - 1)

    return cov_matrix


def compute_corr_matrix(feats):
    cov_matrix = compute_covariance(feats)
    variance_diag = torch.diagonal(cov_matrix)
    zero_v = variance_diag <= 0
    standardize_matrix = torch.diagflat(((variance_diag + zero_v) ** (-0.5)) * (~zero_v))
    corr_matrix = standardize_matrix @ cov_matrix @ standardize_matrix
    return corr_matrix


@register.ood_alg_register
class Subnetwork(BaseOODAlg):
    r"""
    Implementation of the IRM algorithm from `"Invariant Risk Minimization"
    <https://arxiv.org/abs/1907.02893>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Subnetwork, self).__init__(config)
        self.layer_feat = None
        self.graph_feat = None
        self.virtual_node_feat = None
        self.subnetwork_logits = None
        self.model_w0 = None

    def stage_control(self, config: Union[Munch, CommonArgs]):
        self.subnetwork_logits = config.train_helper.model.feature_extractor.encoder.subnetwork_logits
        config.other_saved = {'subnetwork_logits': self.subnetwork_logits}

        if self.stage == 0 and at_stage(1, config):
            # self.model_w0 = deepcopy(config.train_helper.model.state_dict())
            reset_random_seed(config)
            self.stage = 1
        if self.stage == 1 and at_stage(2, config):
            config.train_helper.optimizer = torch.optim.Adam([self.subnetwork_logits],
                                                             lr=1e-1)
            print(f"#IM#\n--------------------- Start stage II ------------------------")
            self.stage = 2
        if self.stage < 3 and at_stage(3, config):

            # model.load_state_dict(self.model_w0)
            model = load_model(config.model.model_name, config)
            config.train_helper.set_up(model, config)

            trained_subnetwork_logits = config.other_saved['subnetwork_logits'].data.clone().detach()
            self.subnetwork_logits = config.train_helper.model.feature_extractor.encoder.subnetwork_logits
            self.subnetwork_logits.data = trained_subnetwork_logits

            reset_random_seed(config)
            print(f"#IM#\n--------------------- Start stage III ------------------------")
            print(f'Mask size: {(self.subnetwork_logits > 0).sum()}')
            self.stage = 3

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model; apply the linear classifier

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions with the linear classifier applied

        """
        self.layer_feat = model_output[1]
        self.graph_feat = model_output[2]
        if len(model_output) > 3:
            self.virtual_node_feat = model_output[3]
        self.raw_pred = model_output[0]
        return model_output[0]

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        r"""
        Process loss based on IRM algorithm

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
            loss with IRM penalty

        """
        if at_stage(2, config):
            spec_loss = config.ood.ood_param * self.subnetwork_logits.sum()
        else:
            spec_loss = 0

        self.mean_loss = loss.sum() / mask.sum()
        self.spec_loss = spec_loss
        total_loss = self.mean_loss + self.spec_loss
        return total_loss

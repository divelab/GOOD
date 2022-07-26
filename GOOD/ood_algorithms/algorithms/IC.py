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
class IC(BaseOODAlg):
    r"""
    Implementation of the IRM algorithm from `"Invariant Risk Minimization"
    <https://arxiv.org/abs/1907.02893>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(IC, self).__init__(config)
        self.layer_feat = None
        self.graph_feat = None
        self.virtual_node_feat = None
        self.stage = 1

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:
        r"""
        Set optimizer.

        Args:
            data (Batch): input data
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            training (bool): whether the task is training
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.ood.ood_param`)

        .. code-block:: python

            config = munchify({device: torch.device('cuda'),
                                   ood: {ood_param: float(0.1)}
                                   })


        Returns:
            - data (Batch) - Processed input data.
            - targets (Tensor) - Processed input labels.
            - mask (Tensor) - Processed NAN masks for data formats.
            - node_norm (Tensor) - Processed node weights for normalization.

        """
        if self.stage < 2 and config.train.epoch >= config.train.stage_stones[0]:
            config.train_helper.optimizer = torch.optim.Adam([config.train_helper.model.feature_selection_logits],
                                                             lr=1e-2)
            print(f"#IM#\n--------------------- Start stage II ------------------------")
            self.stage = 2
        
        # targets = 

        return data, targets, mask, node_norm


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
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss


        return loss

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
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
        model = config.train_helper.model
        metric = config.metric
        if config.train.epoch >= config.train.stage_stones[0]:
            # assert config.train_helper.model.feature_selection_mask is not None
            spec_loss = 1e-4 * config.train_helper.model.feature_selection_logits.sum()
            # spec_loss = 1e-4 * config.train_helper.model.feature_selection_mask.sum()
        else:
            assert self.layer_feat is not None

            indep_weight = 1.
            entropy_weight = 2.
            # decorelate_logits_weight = 1.
            logits_contrastive_weight = 0.e1

            corr_matrix1 = compute_corr_matrix(self.graph_feat[:, :300])
            # corr_matrix2 = compute_corr_matrix(self.graph_feat[:, 300:])
            independent_loss = (corr_matrix1 ** 2).mean()# + (corr_matrix2 ** 2).mean()
            assert not torch.isnan(independent_loss)

            class_indices = [data.y.squeeze() == i for i in range(config.dataset.num_classes)]
            entropy_loss = torch.stack([self.graph_feat[ind].var(dim=0).mean() for ind in class_indices]).sum()
            if torch.isnan(entropy_loss):
                entropy_loss = 0


            # assert not torch.isnan(entropy_loss)

            # logits_corr_matrix = compute_corr_matrix(self.raw_pred)
            # decor_logits_loss = (logits_corr_matrix ** 2).mean()
            #
            tem = 1
            # logits_contrastive_loss = - torch.log(
            #     (torch.exp(
            #         torch.stack([self.raw_pred[ind].mean(dim=0) for ind in class_indices], dim=0).var(0).clamp(
            #             max=10).sum() / tem))
            #     / torch.stack([torch.exp(self.raw_pred[ind].var(dim=0).clamp(max=10).sum() / tem) for ind in
            #                    class_indices]).sum())


            # pos_class_indices = [data.y.squeeze() == i for i in range(config.dataset.num_classes)]
            # neg_class_indices = [data.y.squeeze() != i for i in range(config.dataset.num_classes)]
            # same_class_var = torch.stack([self.raw_pred[ind, i].var() for i, ind in enumerate(pos_class_indices)] + \
            #                  [self.raw_pred[ind, i].var() for i, ind in enumerate(neg_class_indices)], dim=0)
            # contrast_class_mean = torch.stack([self.raw_pred[ind, i].mean() for i, ind in enumerate(pos_class_indices)] + \
            #                       [self.raw_pred[ind, i].mean() for i, ind in enumerate(neg_class_indices)], dim=0)
            # contrast_class_var = contrast_class_mean.reshape(2, config.dataset.num_classes).var(0)
            # logits_contrastive_loss = (same_class_var ** 2).mean() / contrast_class_var.mean()


            # logits_contrastive_loss = - torch.log(
            #     (torch.exp(
            #         torch.stack([self.raw_pred[ind].mean(dim=0) for ind in class_indices], dim=0).var(0).clamp(max=10).sum() / tem))
            #     / torch.stack([torch.exp(self.raw_pred[ind].var(dim=0).clamp(max=10).sum() / tem) for ind in class_indices]).sum())
            # logits_contrastive_loss = - torch.log((torch.exp(self.raw_pred.var(dim=0).clamp(max=10).sum() / tem))/
            #                                       torch.stack([torch.exp(self.raw_pred[ind].var(dim=0).clamp(max=10).sum() / tem) for ind in class_indices]).sum())
            # assert not torch.isinf(logits_contrastive_loss)
            # if torch.isnan(logits_contrastive_loss):
            #     logits_contrastive_loss = 0
            
            # contrast_correct_loss = []
            # contrast_neg_loss = []
            # for i in range(config.dataset.num_classes):
            #     unit_m = torch.diagflat(torch.ones((config.dataset.num_classes,), device=config.device))
            #     targets = unit_m[data.y.squeeze()]
            #     neg_y = - torch.ones((config.dataset.num_classes,), device=config.device)
            #     class_idx = data.y.squeeze() == i
            #     neg_idx = ~class_idx
            #     contrast_correct_loss.append((cross_entropy(model.contrast_out[i][class_idx], targets[class_idx], reduction='none') * mask[class_idx]).sum() / mask[class_idx].sum())
            #     contrast_neg_loss.append((cross_entropy(model.contrast_out[i][neg_idx], neg_y.unsqueeze(0).repeat(neg_idx.sum(), 1), reduction='none') * mask[neg_idx]).sum() / mask[neg_idx].sum())
            # contrast_correct_loss = sum(contrast_correct_loss)
            # contrast_neg_loss = sum(contrast_neg_loss)
            #
            # inner_matrix = model.contrast_selection_mask @ model.contrast_selection_mask.T
            # diag_inner_matrix = torch.diagflat(torch.diagonal(inner_matrix) ** (-0.5))
            # normalize_inner_matrix = diag_inner_matrix @ inner_matrix @ diag_inner_matrix
            # contrast_diverse_loss = normalize_inner_matrix.abs().sum()


            spec_loss = config.ood.ood_param * (
                    indep_weight * independent_loss
                    + entropy_weight * entropy_loss
                    # logits_contrastive_weight * logits_contrastive_loss
                    # + decorelate_logits_weight * decor_logits_loss
                    )
            # assert not torch.isnan(spec_loss)
            # contrast_loss = 1. * contrast_correct_loss + 1.e-2 * contrast_neg_loss + 1.e7 * contrast_diverse_loss
            # if torch.isnan(contrast_loss):
            #     contrast_loss = 0
        mean_loss = loss.sum() / mask.sum()
        self.mean_loss = mean_loss
        self.spec_loss = spec_loss#config.ood.ood_param * entropy_weight * entropy_loss
        loss = self.spec_loss + self.mean_loss
        assert not torch.isnan(loss)
        return loss

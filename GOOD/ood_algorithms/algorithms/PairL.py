"""
Implementation of the Mixup algorithm from `"Mixup for Node and Graph Classification"
<https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper
"""
import copy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
from GOOD.utils.train import at_stage
from GOOD.utils.initial import reset_random_seed


def idNode(data: Batch, id_a2b: Tensor, config: Union[CommonArgs, Munch]) -> Batch:
    r"""
    Mixup node according to given index. Modified from `"MixupForGraph/mixup.py"
    <https://github.com/vanoracai/MixupForGraph/blob/76c2f8b7138b597bdd95a33b0bb32376e3f55227/mixup.py#L46>`_ code.

    Args:
        data (Batch): input data
        id_a2b (Tensor): the random permuted index tensor to index each mixup pair
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`)

    .. code-block:: python

        config = munchify({device: torch.device('cuda')})

    Returns (Batch):
        mixed-up data

    """
    data.x = None
    data.y[data.val_id] = -1
    data.y[data.test_id] = -1
    data.y = data.y[id_a2b]

    data.train_id = None
    data.test_id = None
    data.val_id = None

    id_b2a = torch.zeros(id_a2b.shape[0], dtype=torch.long, device=config.device)
    id_b2a[id_a2b] = torch.arange(0, id_a2b.shape[0], dtype=torch.long, device=config.device)
    row = data.edge_index[0]
    col = data.edge_index[1]
    row = id_b2a[row]
    col = id_b2a[col]
    data.edge_index = torch.stack([row, col], dim=0)

    return data


def shuffleData(data: Batch, config: Union[CommonArgs, Munch]) -> Tuple[Batch, Tensor]:
    r"""
    Prepare data and index for node mixup. Modified from `"MixupForGraph/mixup.py"
    <https://github.com/vanoracai/MixupForGraph/blob/76c2f8b7138b597bdd95a33b0bb32376e3f55227/mixup.py#L46>`_ code.

    Args:
        data (Batch): input data
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`)

    .. code-block:: python

        config = munchify({device: torch.device('cuda')})

    Returns:
        [data (Batch) - mixed-up data,
        id_a2b (Tensor) - the random permuted index tensor to index each mixup pair]

    """
    data = copy.deepcopy(data)
    data.train_id = torch.nonzero(data.train_mask)
    data.val_id = torch.nonzero(data.val_mask)
    data.test_id = torch.nonzero(data.test_mask)
    # --- identify new id by providing old id value ---
    id_a2b = torch.arange(data.num_nodes, device=config.device)
    train_id_shuffle = copy.deepcopy(data.train_id)
    # random.shuffle(train_id_shuffle)
    train_id_shuffle = train_id_shuffle[torch.randperm(train_id_shuffle.shape[0])]
    id_a2b[data.train_id] = train_id_shuffle
    data = idNode(data, id_a2b, config)

    return data, id_a2b


@register.ood_alg_register
class PairL(BaseOODAlg):
    r"""
    Implementation of the Mixup algorithm from `"Mixup for Node and Graph Classification"
    <https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.model.model_level`, :obj:`config.metric.loss_func()`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(PairL, self).__init__(config)
        self.best_pair = [[{'loss': float('inf'), 'data': None} for env_id in range(config.dataset.num_envs)] for target in range(config.dataset.num_classes + 1)]
        # assert config.model.global_pool == 'id'

    def stage_control(self, config: Union[CommonArgs, Munch]):
        if self.stage == 0 and at_stage(1, config):
            config.metric.set_loss_func('Multi-label classification')
            reset_random_seed(config)
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
        Set input data and mask format to prepare for mixup

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
        targets = targets.long().reshape(-1)
        mask = mask.reshape(-1)
        if training:
            # num_data = data.batch[-1] + 1
            # env_ids = torch.zeros((config.dataset.num_envs, num_data), dtype=torch.bool, device=config.device)
            # target_ids = torch.zeros((targets.max() + 1, num_data), dtype=torch.bool, device=config.device)
            # for env_id in data.env_id.unique():
            #     env_ids[env_id] = data.env_id == env_id
            # for target in targets.unique():
            #     target_ids[target] = targets == target
            #
            # orig_data = []
            # pair_data = []
            # is_paired = torch.ones((num_data,), dtype=torch.bool, device=config.device)
            # for i in range(num_data):
            #     graph = data[i]
            #     if training:
            #         select_idx = torch.where(~env_ids[graph.env_id].squeeze() & target_ids[targets[i]])[0]
            #         select_idx = select_idx[0] if select_idx.shape[0] > 0 else None
            #         if select_idx is not None:
            #             mask[i] = mask[i] & mask[select_idx]
            #             orig_data.append(graph)
            #             pair_data.append(data[select_idx])
            #         else:
            #             is_paired[i] = False

            # data = Batch.from_data_list(orig_data + pair_data)
            t1, t2 = targets.chunk(2)
            assert (t1 != t2).sum() == 0

            targets = t1
            mask = mask.chunk(2)[0]
        else:
            # num_data = data.batch[-1] + 1
            # orig_data = []
            # pair_data = []
            # for target in range(config.dataset.num_classes):
            #     for i in range(num_data):
            #         graph = data[i]
            #         for env_id in range(config.dataset.num_envs):
            #             if graph.env_id != env_id:
            #                 orig_data.append(graph)
            #                 pair_data.append(self.best_pair[target][env_id]['data'])
            #                 break
            # data = Batch.from_data_list(orig_data + pair_data)
            pass

        return data, targets, mask, node_norm

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        r"""
        Process loss

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns (Tensor):
            processed loss

        """
        # if config.train_helper.model.training:
        #     for target in range(data.y.max() + 1):
        #         for env_id in range(config.dataset.num_envs):
        #             idx = (data.y.chunk(2)[0] == target) & (data.env_id.chunk(2)[0] == env_id)
        #             if idx.sum() > 0 and loss[idx].min() < self.best_pair[target][env_id]['loss']:
        #                 self.best_pair[target][env_id]['loss'] = loss[idx].min()
        #                 self.best_pair[target][env_id]['data'] = data[idx][loss[idx].argmin()]

        self.mean_loss = loss.sum() / mask.sum()
        return self.mean_loss


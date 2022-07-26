
import copy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


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
class GraphMix(BaseOODAlg):
    r"""
    Implementation of the Mixup algorithm from `"Mixup for Node and Graph Classification"
    <https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.model.model_level`, :obj:`config.metric.loss_func()`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GraphMix, self).__init__(config)
        self.lam = None
        self.data_perm = None
        self.id_a2b: Tensor

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
        self.lam = 0.5
        batch_size = data.batch[-1] + 1
        if training:
            new_batch = []
            org_batch = []
            self.id_a2b = torch.randperm(batch_size)
            for idx_a, idx_b in enumerate(self.id_a2b):
                data_a = data[idx_a]
                data_b = data[idx_b]
                x = torch.cat((data_a.x, data_b.x), dim=0)
                num_bridge = 1
                if config.dataset.dataset_type == 'mol':
                    bridge_attr_idx = torch.randint(0, data_a.edge_attr.shape[0], (1, num_bridge), device=config.device)
                    bridge_attr = torch.squeeze(copy.deepcopy(data_a.edge_attr[bridge_attr_idx]), dim=0)
                    # bridge_attr = torch.zeros((num_bridge, data_a.edge_attr.shape[1]), device=config.device).long()
                    edge_attr = torch.cat((data_a.edge_attr, data_b.edge_attr, bridge_attr), dim=0)
                bridge_a = torch.randint(0, data_a.x.shape[0], (1, num_bridge), device=config.device)
                bridge_b = torch.randint(0, data_b.x.shape[0], (1, num_bridge), device=config.device) + data_a.x.shape[0]
                bridge = torch.cat((bridge_a, bridge_b), dim=0)
                edge_idx = torch.cat((data_a.edge_index, data_b.edge_index + data_a.x.shape[0], bridge), dim=1)
                if config.dataset.dataset_type == 'mol':
                    new_batch.append(Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, y=data_a.y))
                    org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, edge_attr=data_a.edge_attr, y=data_a.y))
                else:
                    new_batch.append(Data(x=x, edge_index=edge_idx, y=data_a.y))
                    org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, y=data_a.y))

            data = Batch.from_data_list(org_batch + new_batch)
            targets = torch.cat((targets, targets))
            mask_mix = mask & mask[self.id_a2b]
            mask = torch.cat((mask, mask_mix))
            self.id_a2b = torch.cat((torch.arange(batch_size), self.id_a2b))
        else:
            self.lam = 1
            self.id_a2b = torch.arange(data.num_nodes, device=config.device)

        return data, targets, mask, node_norm

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss based on Mixup algorithm

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func()}
                                   })


        Returns (Tensor):
            loss based on Mixup algorithm

        """
        loss_a = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss_b = config.metric.loss_func(raw_pred, targets[self.id_a2b], reduction='none') * mask
        if config.model.model_level == 'node':
            loss_a = loss_a * node_norm * mask.sum()
            loss_b = loss_b * node_norm[self.id_a2b] * mask.sum()
        loss = self.lam * loss_a + (1 - self.lam) * loss_b
        return loss

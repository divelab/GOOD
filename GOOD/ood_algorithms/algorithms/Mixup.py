import copy

import numpy as np
import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


def idNode(data, id_a2b, config: Union[CommonArgs, Munch]):
    """
    Modified from https://github.com/vanoracai/MixupForGraph/blob/76c2f8b7138b597bdd95a33b0bb32376e3f55227/mixup.py#L46
    Args:
        data:
        id_a2b:

    Returns:
    :param config:

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


def shuffleData(data, config: Union[CommonArgs, Munch]):
    """
    Modified from https://github.com/vanoracai/MixupForGraph/blob/76c2f8b7138b597bdd95a33b0bb32376e3f55227/mixup.py#L46
    Args:
        data:

    Returns:
    :param config:

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
class Mixup(BaseOODAlg):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Mixup, self).__init__(config)
        self.lam = None
        self.data_perm = None
        self.id_a2b = None

    def input_preprocess(self, data, targets, mask, node_norm, training, config: Union[CommonArgs, Munch], **kwargs):
        if training:
            targets = targets.float()
            alpha = config.ood.ood_param  # 2,4
            self.lam = np.random.beta(alpha, alpha)
            mixup_size = data.y.shape[0]
            self.id_a2b = torch.randperm(mixup_size)
            if node_norm is not None:
                self.data_perm, self.id_a2b = shuffleData(data, config)
            mask = mask & mask[self.id_a2b]
        else:
            self.lam = 1
            self.id_a2b = torch.arange(data.num_nodes, device=config.device)

        return data, targets, mask, node_norm

    def loss_calculate(self, raw_pred, targets, mask, node_norm, config: Union[CommonArgs, Munch]):
        loss_a = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss_b = config.metric.loss_func(raw_pred, targets[self.id_a2b], reduction='none') * mask
        if config.model.model_level == 'node':
            loss_a = loss_a * node_norm * mask.sum()
            loss_b = loss_b * node_norm[self.id_a2b] * mask.sum()
        loss = self.lam * loss_a + (1 - self.lam) * loss_b
        return loss

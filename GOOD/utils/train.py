r"""Training utils.
"""
from typing import Union

import torch
from munch import Munch
from torch_geometric.data import Batch

from GOOD.utils.args import CommonArgs


def nan2zero_get_mask(data, task, config: Union[CommonArgs, Munch]):
    r"""
    Training data filter masks to process NAN.

    Args:
        data (Batch): input data
        task (str): mask function type
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.model_level`)

    Returns (Tensor):
        [mask (Tensor) - NAN masks for data formats, targets (Tensor) - input labels]

    """
    if config.model.model_level == 'node':
        if 'train' in task:
            mask = data.train_mask
        elif task == 'id_val':
            mask = data.get('id_val_mask')
        elif task == 'id_test':
            mask = data.get('id_test_mask')
        elif task == 'val':
            mask = data.val_mask
        elif task == 'test':
            mask = data.test_mask
        else:
            raise ValueError(f'Task should be train/id_val/id_test/val/test, but got {task}.')
    else:
        mask = ~torch.isnan(data.y)
    if mask is None:
        return None, None
    targets = torch.clone(data.y).detach()
    assert mask.shape[0] == targets.shape[0]
    mask = mask.reshape(targets.shape)
    targets[~mask] = 0

    return mask, targets


def at_stage(i, config):
    r"""
    Test if the current training stage at stage i.

    Args:
        i: Stage that is possibly 1, 2, 3, ...
        config: config object.

    Returns: At stage i.

    """
    if i - 1 < 0:
        raise ValueError(f"Stage i must be equal or larger than 0, but got {i}.")
    if i > len(config.train.stage_stones):
        raise ValueError(f"Stage i should be smaller than the largest stage {len(config.train.stage_stones)},"
                         f"but got {i}.")
    if i - 2 < 0:
        return config.train.epoch < config.train.stage_stones[i - 1]
    else:
        return config.train.stage_stones[i - 2] <= config.train.epoch < config.train.stage_stones[i - 1]

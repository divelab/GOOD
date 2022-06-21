r"""An important training utils file for optimizers, schedulers, checkpoint saving, training data filters.
"""
import datetime
import os
import shutil
from typing import Union

import torch
from munch import Munch
from torch_geometric.data import Batch

from GOOD.utils.args import CommonArgs


class TrainHelper(object):
    r"""
    Training utils for optimizers, schedulers, checkpoint saving.
    """

    def __init__(self):
        self.optimizer: torch.optim.Adam = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = None
        self.model: torch.nn.Module = None

    def set_up(self, model, config: Union[CommonArgs, Munch]):
        r"""
        Training setup of optimizer and scheduler

        Args:
            model (dict): model for setup
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.train.lr`, :obj:`config.metric`, :obj:`config.train.mile_stones`)

        Returns:
            None

        """
        self.model: torch.nn.Module = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr,
                                          weight_decay=config.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.train.mile_stones,
                                                              gamma=0.1)

    def save_epoch(self, epoch: int, train_stat: dir, id_val_stat: dir, id_test_stat: dir, val_stat: dir,
                   test_stat: dir, config: Union[CommonArgs, Munch]):
        r"""
        Training util for checkpoint saving.

        Args:
            epoch (int): epoch number
            train_stat (dir): train statistics
            id_val_stat (dir): in-domain validation statistics
            id_test_stat (dir): in-domain test statistics
            val_stat (dir): ood validation statistics
            test_stat (dir): ood test statistics
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.ckpt_dir`, :obj:`config.dataset`, :obj:`config.train`, :obj:`config.model`, :obj:`config.metric`, :obj:`config.log_path`, :obj:`config.ood`)

        Returns:
            None

        """

        ckpt = {
            'state_dict': self.model.state_dict(),
            'train_score': train_stat['score'],
            'train_loss': train_stat['loss'],
            'id_val_score': id_val_stat['score'],
            'id_val_loss': id_val_stat['loss'],
            'id_test_score': id_test_stat['score'],
            'id_test_loss': id_test_stat['loss'],
            'val_score': val_stat['score'],
            'val_loss': val_stat['loss'],
            'test_score': test_stat['score'],
            'test_loss': test_stat['loss'],
            'time': datetime.datetime.now().strftime('%b%d %Hh %M:%S'),
            'model': {
                'model name': f'{config.model.model_name} {config.model.model_level} layers',
                'dim_hidden': config.model.dim_hidden,
                'dim_ffn': config.model.dim_ffn,
                'global pooling': config.model.global_pool
            },
            'dataset': config.dataset.dataset_name,
            'train': {
                'weight_decay': config.train.weight_decay,
                'learning_rate': config.train.lr,
                'mile stone': config.train.mile_stones,
                'shift_type': config.dataset.shift_type,
                'Batch size': f'{config.train.train_bs}, {config.train.val_bs}, {config.train.test_bs}'
            },
            'OOD': {
                'OOD alg': config.ood.ood_alg,
                'OOD param': config.ood.ood_param,
                'number of environments': config.dataset.num_envs
            },
            'log file': config.log_path,
            'epoch': epoch,
            'max epoch': config.train.max_epoch
        }
        if not (config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat['score'] < config.metric.lower_better *
                config.metric.best_stat['score']
                or (id_val_stat.get('score') and (
                        config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
                    'score'] < config.metric.lower_better * config.metric.id_best_stat['score']))
                or epoch % config.train.save_gap == 0):
            return

        if not os.path.exists(config.ckpt_dir):
            os.makedirs(config.ckpt_dir)
            print(f'#W#Directory does not exists. Have built it automatically.\n'
                  f'{os.path.abspath(config.ckpt_dir)}')
        saved_file = os.path.join(config.ckpt_dir, f'{epoch}.ckpt')
        torch.save(ckpt, saved_file)
        shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'last.ckpt'))

        # --- In-Domain checkpoint ---
        if id_val_stat.get('score') and (config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
            'score'] < config.metric.lower_better * config.metric.id_best_stat['score']):
            config.metric.id_best_stat['score'] = id_val_stat['score']
            config.metric.id_best_stat['loss'] = id_val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'id_best.ckpt'))
            print('#IM#Saved a new best In-Domain checkpoint.')

        # --- Out-Of-Domain checkpoint ---
        if id_val_stat.get('score'):
            if not (config.metric.lower_better * id_val_stat['score'] < config.metric.lower_better * val_stat['score']):
                return
        if config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat['score'] < config.metric.lower_better * \
                config.metric.best_stat['score']:
            config.metric.best_stat['score'] = val_stat['score']
            config.metric.best_stat['loss'] = val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'best.ckpt'))
            print('#IM#Saved a new best checkpoint.')


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
    targets[~mask] = 0

    return mask, targets

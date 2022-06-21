r"""Training pipeline: training/evaluation structure, batch training.
"""

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from GOOD.kernel.evaluation import evaluate
from GOOD.networks.model_manager import config_model
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.logger import pbar_setting
from GOOD.utils.train import nan2zero_get_mask


def train_batch(model: torch.nn.Module, data: Batch, ood_algorithm: BaseOODAlg, pbar,
                config: Union[CommonArgs, Munch]) -> dict:
    r"""
    Train a batch. (Project use only)

    Args:
        model (torch.nn.Module): The GNN model.
        data (Batch): Current batch of data.
        ood_algorithm (BaseOODAlg: The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    Returns:
        Calculated loss.
    """
    data = data.to(config.device)

    config.train_helper.optimizer.zero_grad()

    mask, targets = nan2zero_get_mask(data, 'train', config)
    node_norm = data.node_norm if config.model.model_level == 'node' else None
    data, targets, mask, node_norm = ood_algorithm.input_preprocess(data, targets, mask, node_norm, model.training,
                                                                    config)
    edge_weight = data.edge_norm if config.model.model_level == 'node' else None

    model_output = model(data=data, edge_weight=edge_weight, ood_algorithm=ood_algorithm)
    raw_pred = ood_algorithm.output_postprocess(model_output)

    loss = ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, config)
    loss = ood_algorithm.loss_postprocess(loss, data, mask, config)
    loss.backward()

    config.train_helper.optimizer.step()

    return {'loss': loss.detach()}


def train(model: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]], ood_algorithm: BaseOODAlg,
          config: Union[CommonArgs, Munch]):
    r"""
    Training pipeline. (Project use only)

    Args:
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    """
    # config model
    print('#D#Config model')
    config_model(model, 'train', config)

    # Load training utils
    print('#D#Load training utils')
    config.train_helper.set_up(model, config)

    # train the model
    for epoch in range(config.train.ctn_epoch, config.train.max_epoch):

        print(f'#IN#Epoch {epoch}:')

        mean_loss = 0
        spec_loss = 0

        pbar = tqdm(enumerate(loader['train']), total=len(loader['train']), **pbar_setting)
        for index, data in pbar:
            if data.batch is not None and (data.batch[-1] < config.train.train_bs - 1):
                continue

            # Parameter for DANN
            p = (index / len(loader['train']) + epoch) / config.train.max_epoch
            config.train.alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # train a batch
            train_stat = train_batch(model, data, ood_algorithm, pbar, config)
            mean_loss = (mean_loss * index + ood_algorithm.mean_loss) / (index + 1)

            if config.ood.ood_alg not in ['ERM', 'GroupDRO', 'Mixup']:
                spec_loss = (spec_loss * index + ood_algorithm.spec_loss) / (index + 1)
                pbar.set_description(f'M/S Loss: {mean_loss:.4f}/{spec_loss:.4f}')
            else:
                pbar.set_description(f'Loss: {mean_loss:.4f}')

        # Eval training score

        # Epoch val
        print('#IN#\nEvaluating...')
        if config.ood.ood_alg not in ['ERM', 'GroupDRO', 'Mixup']:
            print(f'#IN#Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}')
        else:
            print(f'#IN#Approximated average training loss {mean_loss.cpu().item():.4f}')

        epoch_train_stat = evaluate(model, loader, ood_algorithm, 'eval_train', config)
        id_val_stat = evaluate(model, loader, ood_algorithm, 'id_val', config)
        id_test_stat = evaluate(model, loader, ood_algorithm, 'id_test', config)
        val_stat = evaluate(model, loader, ood_algorithm, 'val', config)
        test_stat = evaluate(model, loader, ood_algorithm, 'test', config)

        # checkpoints save
        config.train_helper.save_epoch(epoch, epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat, config)

        # --- scheduler step ---
        config.train_helper.scheduler.step()

    print('#IN#Training end.')

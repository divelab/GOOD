r"""Training pipeline: training/evaluation structure, batch training.
"""
import os
from typing import Dict

import numpy as np
import torch
import torch.nn
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.logger import pbar_setting
from GOOD.utils.train import nan2zero_get_mask
from GOOD.utils.register import register


@register.pipeline_register
class Pipeline:
    r"""
    Pipeline. (Project use only)

    Args:
        task (str): Current running task. 'train' or 'test'
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    """

    def __init__(self, task: str, model: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]], ood_algorithm: BaseOODAlg,
                  config: Union[CommonArgs, Munch]):
        super(Pipeline, self).__init__()
        self.task: str = task
        self.model: torch.nn.Module = model
        self.loader: Union[DataLoader, Dict[str, DataLoader]] = loader
        self.ood_algorithm: BaseOODAlg = ood_algorithm
        self.config: Union[CommonArgs, Munch] = config


    def train_batch(self, data: Batch, pbar) -> dict:
        r"""
        Train a batch. (Project use only)

        Args:
            data (Batch): Current batch of data.

        Returns:
            Calculated loss.
        """
        data = data.to(self.config.device)

        self.config.train_helper.optimizer.zero_grad()

        mask, targets = nan2zero_get_mask(data, 'train', self.config)
        node_norm = data.get('node_norm') if self.config.model.model_level == 'node' else None
        data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm, self.model.training,
                                                                        self.config)
        edge_weight = data.get('edge_norm') if self.config.model.model_level == 'node' else None

        model_output = self.model(data=data, edge_weight=edge_weight, ood_algorithm=self.ood_algorithm)
        raw_pred = self.ood_algorithm.output_postprocess(model_output)

        loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, self.config)
        loss = self.ood_algorithm.loss_postprocess(loss, data, mask, self.config)
        loss.backward()

        self.config.train_helper.optimizer.step()

        return {'loss': loss.detach()}


    def train(self):
        r"""
        Training pipeline. (Project use only)
        """
        # config model
        print('#D#Config model')
        self.config_model('train')

        # Load training utils
        print('#D#Load training utils')
        self.config.train_helper.set_up(self.model, self.config)

        # train the model
        for epoch in range(self.config.train.ctn_epoch, self.config.train.max_epoch):
            self.config.train.epoch = epoch
            print(f'#IN#Epoch {epoch}:')

            mean_loss = 0
            spec_loss = 0

            self.ood_algorithm.stage_control(self.config)

            pbar = tqdm(enumerate(self.loader['train']), total=len(self.loader['train']), **pbar_setting)
            for index, data in pbar:
                if data.batch is not None and (data.batch[-1] < self.config.train.train_bs - 1):
                    continue

                # Parameter for DANN
                p = (index / len(self.loader['train']) + epoch) / self.config.train.max_epoch
                self.config.train.alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # train a batch
                train_stat = self.train_batch(data, pbar)
                mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (index + 1)

                if self.config.ood.ood_alg not in ['ERM', 'GroupDRO', 'Mixup']:
                    spec_loss = (spec_loss * index + self.ood_algorithm.spec_loss) / (index + 1)
                    pbar.set_description(f'M/S Loss: {mean_loss:.4f}/{spec_loss:.4f}')
                else:
                    pbar.set_description(f'Loss: {mean_loss:.4f}')

            # Eval training score

            # Epoch val
            print('#IN#\nEvaluating...')
            if self.config.ood.ood_alg not in ['ERM', 'GroupDRO', 'Mixup']:
                print(f'#IN#Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}')
            else:
                print(f'#IN#Approximated average training loss {mean_loss.cpu().item():.4f}')

            epoch_train_stat = self.evaluate('eval_train')
            id_val_stat = self.evaluate('id_val')
            id_test_stat = self.evaluate('id_test')
            val_stat = self.evaluate('val')
            test_stat = self.evaluate('test')

            # checkpoints save
            self.config.train_helper.save_epoch(epoch, epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat, self.config)

            # --- scheduler step ---
            self.config.train_helper.scheduler.step()

        print('#IN#Training end.')

    @torch.no_grad()
    def evaluate(self, split: str):
        r"""
        This function is design to collect data results and calculate scores and loss given a dataset subset.
        (For project use only)

        Args:
            split (str): A split string for choosing the corresponding dataloader. Allowed: 'train', 'id_val', 'id_test',
                'val', and 'test'.

        Returns:
            A score and a loss.

        """
        stat = {'score': None, 'loss': None}
        if self.loader.get(split) is None:
            return stat
        self.model.eval()

        loss_all = []
        mask_all = []
        pred_all = []
        target_all = []
        pbar = tqdm(self.loader[split], desc=f'Eval {split.capitalize()}', total=len(self.loader[split]),
                    **pbar_setting)
        for data in pbar:
            data: Batch = data.to(self.config.device)

            mask, targets = nan2zero_get_mask(data, split, self.config)
            if mask is None:
                return stat
            node_norm = torch.ones((data.num_nodes,),
                                   device=self.config.device) if self.config.model.model_level == 'node' else None
            data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                                 self.model.training,
                                                                                 self.config)
            model_output = self.model(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            raw_preds = self.ood_algorithm.output_postprocess(model_output)

            # --------------- Loss collection ------------------
            loss: torch.tensor = self.config.metric.loss_func(raw_preds, targets, reduction='none') * mask
            mask_all.append(mask)
            loss_all.append(loss)

            # ------------- Score data collection ------------------
            pred, target = eval_data_preprocess(data.y, raw_preds, mask, self.config)
            pred_all.append(pred)
            target_all.append(target)

        # ------- Loss calculate -------
        loss_all = torch.cat(loss_all)
        mask_all = torch.cat(mask_all)
        stat['loss'] = loss_all.sum() / mask_all.sum()

        # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
        stat['score'] = eval_score(pred_all, target_all, self.config)

        print(f'#IN#\n{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f}\n'
              f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

        self.model.train()

        return {'score': stat['score'], 'loss': stat['loss']}


    def load_task(self):
        r"""
        Launch a training or a test. (Project use only)
        """
        if self.task == 'train':
            self.train()

        elif self.task == 'test':

            # config model
            print('#D#Config model and output the best checkpoint info...')
            test_score, test_loss = self.config_model('test')


    def config_model(self, mode: str, load_param=False):
        r"""
        A model configuration utility. Responsible for transiting model from CPU -> GPU and loading checkpoints.
        Args:
            mode (str): 'train' or 'test'.
            load_param: When True, loading test checkpoint will load parameters to the GNN model.

        Returns:
            Test score and loss if mode=='test'.
        """
        self.model.to(self.config.device)
        self.model.train()

        # load checkpoint
        if mode == 'train' and self.config.train.tr_ctn:
            ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'last.ckpt'))
            self.model.load_state_dict(ckpt['state_dict'])
            best_ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'best.ckpt'))
            self.config.metric.best_stat['score'] = best_ckpt['val_score']
            self.config.metric.best_stat['loss'] = best_ckpt['val_loss']
            self.config.train.ctn_epoch = ckpt['epoch'] + 1
            print(f'#IN#Continue training from Epoch {ckpt["epoch"]}...')

        if mode == 'test':
            try:
                ckpt = torch.load(self.config.test_ckpt, map_location=self.config.device)
            except FileNotFoundError:
                print(f'#E#Checkpoint not found at {os.path.abspath(self.config.test_ckpt)}')
                exit(1)
            if os.path.exists(self.config.id_test_ckpt):
                id_ckpt = torch.load(self.config.id_test_ckpt, map_location=self.config.device)
                # model.load_state_dict(id_ckpt['state_dict'])
                print(f'#IN#Loading best In-Domain Checkpoint {id_ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {id_ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {id_ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {id_ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {id_ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {id_ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {id_ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {id_ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {id_ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {id_ckpt["val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {id_ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {id_ckpt["test_loss"].item():.4f}\n')
                print(f'#IN#Loading best Out-of-Domain Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {ckpt["test_loss"].item():.4f}\n')

                print(f'#IN#ChartInfo {id_ckpt["id_test_score"]:.4f} {id_ckpt["test_score"]:.4f} '
                      f'{ckpt["id_test_score"]:.4f} {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')

            else:
                print(f'#IN#No In-Domain checkpoint.')
                # model.load_state_dict(ckpt['state_dict'])
                print(f'#IN#Loading best Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                      f'Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'Test Loss: {ckpt["test_loss"].item():.4f}\n')

                print(
                    f'#IN#ChartInfo {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')
            if load_param:
                if self.config.ood.ood_alg != 'EERM':
                    self.model.load_state_dict(ckpt['state_dict'])
                else:
                    self.model.gnn.load_state_dict(ckpt['state_dict'])
            return ckpt["test_score"], ckpt["test_loss"]
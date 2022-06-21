r"""
Evaluation: model evaluation functions.
"""

import numpy as np
import torch
from torch_geometric.data.batch import Batch
from torch.utils.data import DataLoader
from tqdm import tqdm

from typing import List, Dict, Tuple
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.logger import pbar_setting
from GOOD.utils.train import nan2zero_get_mask
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg


def eval_data_preprocess(y: torch.Tensor,
                         raw_pred: torch.Tensor,
                         mask: torch.Tensor,
                         config: Union[CommonArgs, Munch]
                         ) -> Tuple[Union[np.ndarray, List], Union[np.ndarray, List]]:
    r"""
    Preprocess data for evaluations by converting data into np.ndarray or List[np.ndarray] (Multi-task) format.
    When the task of the dataset is not multi-task, data is converted into np.ndarray.
    When it is multi-task, data is converted into List[np.ndarray] in which each np.ndarray in the list represents
    one task. For example, GOOD-PCBA is a 128-task binary classification dataset. Therefore, the output list will
    contain 128 elements.

    Args:
        y (torch.Tensor): Ground truth values.
        raw_pred (torch.Tensor): Raw prediction values without softmax or sigmoid.
        mask (torch.Tensor): Ground truth NAN mask for removing empty label.
        config (Union[CommonArgs, Munch]): The required config is
            ``config.metric.dataset_task``

    Returns:
        Processed prediction values and ground truth values.

    """
    if config.metric.dataset_task == 'Binary classification':
        pred_prob = raw_pred.sigmoid()
        if y.shape[1] > 1:
            # multi-task
            preds = []
            targets = []
            for i in range(y.shape[1]):
                # pred and target per task
                preds.append(pred_prob[:, i][mask[:, i]].detach().cpu().numpy())
                targets.append(y[:, i][mask[:, i]].detach().cpu().numpy())
            return preds, targets
        pred = pred_prob[mask].reshape(-1).detach().cpu().numpy()
    elif config.metric.dataset_task == 'Multi-label classification':
        pred_prob = raw_pred.softmax(dim=1)
        pred = pred_prob[mask].detach().cpu().numpy()
    elif 'Regression' in config.metric.dataset_task:
        pred = raw_pred[mask].reshape(-1).detach().cpu().numpy()
    else:
        raise ValueError('Dataset task value error.')

    target = y[mask].reshape(-1).detach().cpu().numpy()

    return pred, target


def eval_score(pred_all: Union[List[np.ndarray], List[List[np.ndarray]]],
               target_all: Union[List[np.ndarray], List[List[np.ndarray]]],
               config: Union[CommonArgs, Munch]
               ) -> Union[np.ndarray, np.float, float]:
    r"""
    Calculate metric scores given preprocessed prediction values and ground truth values.

    Args:
        pred_all (Union[List[np.ndarray], List[List[np.ndarray]]]): Prediction value list. It is a list of output pred
            of :func:`eval_data_preprocess`.
        target_all (Union[List[np.ndarray], List[List[np.ndarray]]]): Ground truth value list. It is a list of output
            target of :func:`eval_data_preprocess`.
        config (Union[CommonArgs, Munch]): The required config is ``config.metric.score_func`` that is a function for
            score calculation (*e.g.*, :func:`GOOD.utils.metric.Metric.acc`).

    Returns:
        A float score value.
    """
    np.seterr(invalid='ignore')

    if type(pred_all) is list:
        if type(pred_all[0]) is list:
            # multi-task
            all_task_preds = []
            all_task_targets = []
            for task_i in range(len(pred_all[0])):
                preds = []
                targets = []
                for pred, target in zip(pred_all, target_all):
                    preds.append(pred[task_i])
                    targets.append(target[task_i])
                all_task_preds.append(np.concatenate(preds))
                all_task_targets.append(np.concatenate(targets))

            scores = []
            for i in range(len(all_task_preds)):
                if all_task_targets[i].shape[0] > 0:
                    scores.append(np.nanmean(config.metric.score_func(all_task_targets[i], all_task_preds[i])))
            score = np.nanmean(scores)
        else:
            pred_all = np.concatenate(pred_all)
            target_all = np.concatenate(target_all)
            score = np.nanmean(config.metric.score_func(target_all, pred_all))
    else:
        raise ValueError('Wrong prediction input.')
    return score


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: Dict[str, DataLoader],
             ood_algorithm: BaseOODAlg,
             split: str,
             config: Union[CommonArgs, Munch]
             ):
    r"""
    This function is design to collect data results and calculate scores and loss given a dataset subset.
    (For project use only)

    Args:
        model (torch.nn.Module): The GNN model.
        loader (Dict[str, DataLoader]): A DataLoader dictionary that use ``split`` as key and Dataloaders as values.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        split (str): A split string for choosing the corresponding dataloader. Allowed: 'train', 'id_val', 'id_test',
            'val', and 'test'.
        config (Union[CommonArgs, Munch]): Required configs are ``config.device`` (torch.device),
            ``config.model.model_level``, ``config.metric`` (:class:`GOOD.utils.metric.Metric`),
            ``config.ood`` (:class:`GOOD.utils.args.OODArgs`). Refer to :ref:`configs:Config file` for more details.

    Returns:
        A score and a loss.

    """
    stat = {'score': None, 'loss': None}
    if loader.get(split) is None:
        return stat
    model.eval()

    loss_all = []
    mask_all = []
    pred_all = []
    target_all = []
    pbar = tqdm(loader[split], desc=f'Eval {split.capitalize()}', total=len(loader[split]), **pbar_setting)
    for data in pbar:
        data: Batch = data.to(config.device)

        mask, targets = nan2zero_get_mask(data, split, config)
        if mask is None:
            return stat
        node_norm = torch.ones((data.num_nodes,), device=config.device) if config.model.model_level == 'node' else None
        data, targets, mask, node_norm = ood_algorithm.input_preprocess(data, targets, mask, node_norm, model.training,
                                                                        config)
        model_output = model(data=data, edge_weight=None, ood_algorithm=ood_algorithm)
        raw_preds = ood_algorithm.output_postprocess(model_output)

        # --------------- Loss collection ------------------
        loss: torch.tensor = config.metric.loss_func(raw_preds, targets, reduction='none') * mask
        mask_all.append(mask)
        loss_all.append(loss)

        # ------------- Score data collection ------------------
        pred, target = eval_data_preprocess(data.y, raw_preds, mask, config)
        pred_all.append(pred)
        target_all.append(target)

    # ------- Loss calculate -------
    loss_all = torch.cat(loss_all)
    mask_all = torch.cat(mask_all)
    stat['loss'] = loss_all.sum() / mask_all.sum()

    # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
    stat['score'] = eval_score(pred_all, target_all, config)

    print(f'#IN#\n{split.capitalize()} {config.metric.score_name}: {stat["score"]:.4f}\n'
          f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

    model.train()

    return {'score': stat['score'], 'loss': stat['loss']}

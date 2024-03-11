r"""
Evaluation: model evaluation functions.
"""

import numpy as np
import torch

from typing import List, Tuple
from GOOD.utils.config_reader import Union, CommonArgs, Munch


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
               ) -> Union[np.ndarray, float, float]:
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

    assert type(pred_all) is list, 'Wrong prediction input.'
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
    return score



r"""
Evaluation: model evaluation functions.
"""

import numpy as np
import torch
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.logger import pbar_setting
from GOOD.utils.train import nan2zero_get_mask


def eval_data_preprocess(y, raw_pred, mask, config):
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

    target = y[mask].reshape(-1).detach().cpu().numpy()

    return pred, target


def eval_score(pred_all, target_all, train, config):
    np.seterr(invalid='ignore')
    if train:
        pred = pred_all
        target = target_all
        try:
            if type(pred) is list:
                scores = []
                for i in range(len(pred)):
                    if np.sum(target[i] == 1) > 0 and np.sum(target[i] == 0) > 0:
                        tmp_score = config.metric.score_func(target[i], pred[i])
                        scores.append(np.mean(tmp_score))
                score = np.nanmean(scores)
            else:
                score = np.nanmean(config.metric.score_func(target, pred))
        except ValueError as e:
            print(f"#W#{e}")
            score = 0
    else:
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
    return score


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, ood_algorithm, split, config: Union[CommonArgs, Munch]):
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
    stat['score'] = eval_score(pred_all, target_all, False, config)

    print(f'#IN#\n{split.capitalize()} {config.metric.score_name}: {stat["score"]:.4f}\n'
          f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

    model.train()

    return {'score': stat['score'], 'loss': stat['loss']}

import torch
from torch.autograd import grad
from torch import Tensor
from torch_geometric.data import Batch, Data
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
from typing import Tuple
import numpy as np
from GOOD.utils.train import nan2zero_get_mask


def get_xinv_mask(data, task, config: Union[CommonArgs, Munch]):
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


@register.ood_alg_register
class feat_aug(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(feat_aug, self).__init__(config)
        self.xinv_mask = None
        self.xvar_mask = None
        self.xinv_var = None
        self.w_env = 1.0
        self.w_label = 1.0
        self.thd = 0.0
        self.lam = None

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:

        if config.model.model_level == 'node':
            batch_size = data.train_mask.shape[0]
        else:
            batch_size = int(data.batch[-1]) + 1
        num_class = torch.unique(data.y).shape[0]
        if training and config.model.model_level != 'node':
            if self.xinv_var == None:
                self.xinv_var = torch.zeros(data.x.shape[1], device=config.device)
                self.xinv_mask = torch.zeros(data.x.shape[1], device=config.device)
                self.xvar_mask = torch.ones(data.x.shape[1], device=config.device)
            new_batch = []
            org_batch = []

            for i in range(config.dataset.num_envs):
                env_idx = (data.env_id == i).clone().detach()
                if data.y[env_idx].shape[0] > 0:
                    self.xinv_var = self.xinv_var + self.w_env * torch.var(Batch.from_data_list(data[env_idx]).x, 0)

            for i in range(num_class):
                label_idx = (data.y == i).clone().detach()
                if data.y[label_idx].shape[0] > 0:
                    self.xinv_var = self.xinv_var - self.w_label * torch.var(Batch.from_data_list(data[label_idx]).x, 0)

            self.xinv_mask = (torch.div(self.xinv_var, torch.mean(data.x, 0)) > self.thd).long()
            self.xvar_mask = self.xvar_mask - self.xinv_mask

            data_a = None
            data_b = None

            for i in range(int(batch_size/2)):
                labl0 = np.random.randint(0, num_class)
                for j in range(num_class):
                    labl = (labl0+j)%num_class
                    label_idx = (data.y == labl).clone().detach()
                    if torch.unique(data.env_id[label_idx]).shape[0] < 2:
                        continue
                    else:
                        perm = torch.randperm(torch.unique(data.env_id[label_idx]).shape[0])
                        id1 = torch.unique(data.env_id[label_idx])[perm[0]]
                        id2 = torch.unique(data.env_id[label_idx])[perm[1]]
                        idx_a = torch.logical_and(data.y == labl, data.env_id == id1)
                        idx_b = torch.logical_and(data.y == labl, data.env_id == id2)
                        data_a = data[idx_a][np.random.randint(0,data[idx_a].__len__())]
                        data_b = data[idx_b][np.random.randint(0,data[idx_b].__len__())]
                        break

                if data_a == None:
                    return data, targets, mask, node_norm
                else:
                    self.lam = np.random.beta(1, 1)
                    xvar1 = torch.mean(data_a.x, 0).expand(data_b.x.shape[0], -1)
                    xvar2 = torch.mean(data_b.x, 0).expand(data_a.x.shape[0], -1)
                    x_inter = torch.mul(data_a.x, self.xinv_mask) + torch.mul(self.lam * data_a.x + (1 - self.lam) * xvar2, self.xvar_mask)
                    x_extra1 = torch.mul(data_a.x, self.xinv_mask) + torch.mul(2 * data_a.x - xvar2, self.xvar_mask)
                    x_extra2 = torch.mul(data_b.x, self.xinv_mask) + torch.mul(2 * data_b.x - xvar1, self.xvar_mask)
                    new_batch.append(Data(x=x_inter, edge_index=data_a.edge_index, y=data_a.y))
                    new_batch.append(Data(x=x_extra1, edge_index=data_a.edge_index, y=data_a.y))
                    new_batch.append(Data(x=x_extra2, edge_index=data_b.edge_index, y=data_a.y))
                    org_batch.append(Data(x=data[i].x, edge_index=data[i].edge_index, y=data[i].y))
                    org_batch.append(Data(x=data[i+int(batch_size/2)].x, edge_index=data[i+int(batch_size/2)].edge_index, y=data[i+int(batch_size/2)].y))


            data = Batch.from_data_list(org_batch + new_batch)
            mask, targets = nan2zero_get_mask(data, 'train', config)

        elif training and config.model.model_level == 'node':
            self.w_env = 10.0/config.dataset.num_envs
            self.w_label = 10.0/num_class
            self.thd = -1.0
            if self.xinv_var == None:
                self.xinv_var = torch.zeros(data.x.shape[1], device=config.device)
                self.xinv_mask = torch.zeros(data.x.shape[1], device=config.device)
                self.xvar_mask = torch.ones(data.x.shape[1], device=config.device)
            new_batch = []
            org_batch = []

            for i in range(config.dataset.num_envs):
                env_idx = (data.env_id == i).clone().detach()
                if data.x[env_idx].shape[0] > 0:
                    self.xinv_var = self.xinv_var + self.w_env * torch.var(data.x[env_idx], 0)

            for i in range(num_class):
                label_idx = torch.logical_and(data.y == i, data.train_mask).clone().detach()
                if data.x[label_idx].shape[0] > 0:
                    self.xinv_var = self.xinv_var - self.w_label * torch.var(data.x[label_idx], 0)

            self.xinv_mask = (torch.div(self.xinv_var, torch.mean(data.x, 0)) > self.thd).long()
            self.xvar_mask = self.xvar_mask - self.xinv_mask

            # data_a = None
            # data_b = None
            for i in range(num_class):
                label_idx = torch.logical_and(data.y == i, data.train_mask).clone().detach()
                if torch.unique(data.env_id[label_idx]).shape[0] < 2:
                    continue
                else:
                    in_num = label_idx.nonzero().shape[0]
                    for j in range(int(in_num/2)):
                        stg = np.random.randint(0, 3)
                        idx_a = label_idx.nonzero()[j]
                        valid_b = torch.logical_and(data.env_id != data.env_id[idx_a], label_idx !=0).nonzero().view(-1)
                        choice = torch.multinomial(valid_b.float(), 1)
                        idx_b = valid_b[choice]
                        if stg == 1:
                            data.x[idx_a] = torch.mul(data.x[idx_a], self.xinv_mask) + torch.mul(2 * data.x[idx_a] - data.x[idx_b], self.xvar_mask)
                        elif stg == 2:
                            data.x[idx_a] = torch.mul(data.x[idx_a], self.xinv_mask) + torch.mul(2 * data.x[idx_b] - data.x[idx_a], self.xvar_mask)
                        else:
                            self.lam = np.random.beta(1, 1)
                            data.x[idx_a] = torch.mul(data.x[idx_a], self.xinv_mask) + torch.mul(self.lam * data.x[idx_a] + (1 - self.lam) * data.x[idx_b], self.xvar_mask)




        return data, targets, mask, node_norm

    # def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
    #
    #     raw_pred = self.dummy_w * model_output
    #     return raw_pred
    #
    # def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
    #
    #     spec_loss_list = []
    #     for i in range(config.dataset.num_envs):
    #         env_idx = data.env_id == i
    #         if loss[env_idx].shape[0] > 0:
    #             grad_all = torch.sum(
    #                 grad(loss[env_idx].sum() / mask[env_idx].sum(), self.dummy_w, create_graph=True)[0].pow(2))
    #             spec_loss_list.append(grad_all)
    #     spec_loss = config.ood.ood_param * sum(spec_loss_list) / len(spec_loss_list)
    #     if torch.isnan(spec_loss):
    #         spec_loss = 0
    #     mean_loss = loss.sum() / mask.sum()
    #     loss = spec_loss + mean_loss
    #     self.mean_loss = mean_loss
    #     self.spec_loss = spec_loss
    #     return loss

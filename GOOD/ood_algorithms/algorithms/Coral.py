import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


def compute_covariance(input_data, config: Union[CommonArgs, Munch]):
    """
    Compute Covariance matrix of the input data
    :param input_data:
    :param config:
    """
    n = input_data.shape[0]  # batch_size

    id_row = torch.ones((1, n), device=config.device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c


@register.ood_alg_register
class Coral(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Coral, self).__init__(config)
        self.feat = None

    def output_postprocess(self, model_output, **kwargs):
        self.feat = model_output[1]
        return model_output[0]

    def loss_postprocess(self, loss, data, mask, config: Union[CommonArgs, Munch], **kwargs):
        """
        Thdskdjslf

        Args:
            loss:
            data:
            mask:
            config:
            **kwargs:

        Returns:

        """
        loss_list = []
        covariance_matrices = []
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            env_feat = self.feat[env_idx]
            if env_feat.shape[0] > 1:
                covariance_matrices.append(compute_covariance(env_feat, config))
            else:
                covariance_matrices.append(None)

        for i in range(config.dataset.num_envs):
            for j in range(config.dataset.num_envs):
                if i != j and covariance_matrices[i] is not None and covariance_matrices[j] is not None:
                    dis = covariance_matrices[i] - covariance_matrices[j]
                    cov_loss = torch.mean(torch.mul(dis, dis)) / 4
                    loss_list.append(cov_loss)

        if len(loss_list) == 0:
            coral_loss = torch.tensor(0)
        else:
            coral_loss = sum(loss_list) / len(loss_list)
        spec_loss = config.ood.ood_param * coral_loss
        if torch.isnan(spec_loss):
            spec_loss = 0
        mean_loss = loss.sum() / mask.sum()
        loss = mean_loss + spec_loss
        self.mean_loss = mean_loss
        self.spec_loss = spec_loss
        return loss

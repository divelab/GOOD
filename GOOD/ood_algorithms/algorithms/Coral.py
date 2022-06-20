"""
Implementation of the Deep Coral algorithm from `"Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
<https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35>`_ paper
"""
import torch
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


def compute_covariance(input_data: Tensor, config: Union[CommonArgs, Munch]) -> Tensor:
    r"""
    Compute Covariance matrix of the input data

    Args:
        input_data (Tensor): feature of the input data
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`)

    .. code-block:: python

        config = munchify({device: torch.device('cuda')})

    Returns (Tensor):
        covariance value of the input features

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
    r"""
    Implementation of the Deep Coral algorithm from `"Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
    <https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Coral, self).__init__(config)
        self.feat = None

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model; get feature representations

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        self.feat = model_output[1]
        return model_output[0]

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on Deep Coral algorithm

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)

        .. code-block:: python

            config = munchify({device: torch.device('cuda'),
                                   dataset: {num_envs: int(10)},
                                   ood: {ood_param: float(0.1)}
                                   })


        Returns (Tensor):
            loss based on Deep Coral algorithm

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

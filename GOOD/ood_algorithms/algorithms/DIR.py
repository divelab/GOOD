"""
Implementation of the IRM algorithm from `"Invariant Risk Minimization"
<https://arxiv.org/abs/1907.02893>`_ paper
"""
import torch
from torch.autograd import grad
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


@register.ood_alg_register
class DIR(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DIR, self).__init__(config)
        self.dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(config.device)
        # self.causal_edge_weights = torch.tensor([]).to(config.device)
        # self.conf_edge_weights = torch.tensor([]).to(config.device)

        self.all_loss, self.all_env_loss, self.all_causal_loss = [], [], []

    def stage_control(self, config: Union[CommonArgs, Munch]):
        self.optimizer: torch.optim.Adam([{'params': list(g.parameters()) +
                                                   list(att_net.parameters()),
                                         'lr': config.train.lr},
                                        {'params': g.conf_lin.parameters(),
                                         'lr': config.train.lr}])

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:
        r"""
        Set input data and mask format to prepare for mixup

        Args:
            data (Batch): input data
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            training (bool): whether the task is training
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.ood.ood_param`)

        .. code-block:: python

            config = munchify({device: torch.device('cuda'),
                                   ood: {ood_param: float(0.1)}
                                   })


        Returns:
            - data (Batch) - Processed input data.
            - targets (Tensor) - Processed input labels.
            - mask (Tensor) - Processed NAN masks for data formats.
            - node_norm (Tensor) - Processed node weights for normalization.

        """
        # if training:
        #     targets = targets.float()
        #     alpha = config.ood.ood_param  # 2,4
        #     self.lam = np.random.beta(alpha, alpha)
        #     mixup_size = data.y.shape[0]
        #     self.id_a2b = torch.randperm(mixup_size)
        #     if node_norm is not None:
        #         self.data_perm, self.id_a2b = shuffleData(data, config)
        #     mask = mask & mask[self.id_a2b]
        # else:
        #     self.lam = 1
        #     self.id_a2b = torch.arange(data.num_nodes, device=config.device)

        return data, targets, mask, node_norm

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss based on Mixup algorithm

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func()}
                                   })


        Returns (Tensor):
            loss based on Mixup algorithm

        """
        BCELoss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        # loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        # loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss

        rep_out = raw_pred[0]
        causal_out = raw_pred[1]
        conf_out = raw_pred[2]

        is_labeled = targets == targets
        causal_loss = BCELoss(
            causal_out.to(torch.float32)[is_labeled],
            targets.to(torch.float32)[is_labeled]
        )* mask
        conf_loss = BCELoss(
            conf_out.to(torch.float32)[is_labeled],
            targets.to(torch.float32)[is_labeled]
        )
        env_loss = 0
        # if args.reg:
        env_loss = torch.tensor([]).to(config.device)
        for rep in rep_out:
            tmp = BCELoss(rep.to(torch.float32)[is_labeled], targets.to(torch.float32)[is_labeled])
            env_loss = torch.cat([env_loss, tmp.unsqueeze(0)])
        causal_loss += config.train.alpha * env_loss.mean()
        env_loss = config.train.alpha * torch.var(env_loss * rep_out.size(0))

        # logger
        all_causal_loss.append(causal_loss)
        all_env_loss.append(env_loss)
        # batch_loss = causal_loss + env_loss
        # causal_edge_weights = torch.cat([causal_edge_weights, causal_edge_weight])
        # conf_edge_weights = torch.cat([conf_edge_weights, conf_edge_weight])

        # conf_opt.zero_grad()
        # conf_loss.backward()
        # conf_opt.step()
        #
        # model_optimizer.zero_grad()
        # (causal_loss + env_loss).backward()
        # model_optimizer.step()

        # all_env_loss /= n_bw
        # all_causal_loss /= n_bw
        loss = causal_loss + env_loss + conf_loss
        self.mean_loss = all_causal_loss.mean()
        self.spec_loss = all_env_loss.mean()

        return loss


    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on IRM algorithm

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
            loss with IRM penalty

        """


        # spec_loss_list = []
        # for i in range(config.dataset.num_envs):
        #     env_idx = data.env_id == i
        #     if loss[env_idx].shape[0] > 0:
        #         grad_all = torch.sum(
        #             grad(loss[env_idx].sum() / mask[env_idx].sum(), self.dummy_w, create_graph=True)[0].pow(2))
        #         spec_loss_list.append(grad_all)
        # spec_loss = config.ood.ood_param * sum(spec_loss_list) / len(spec_loss_list)
        # if torch.isnan(spec_loss):
        #     spec_loss = 0
        # mean_loss = loss.sum() / mask.sum()
        # loss = spec_loss + mean_loss
        # self.mean_loss = mean_loss
        # self.spec_loss = spec_loss
        return loss

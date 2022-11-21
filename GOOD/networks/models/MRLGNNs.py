# r"""
# Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism <https://arxiv.org/abs/2201.12987>`_.
# """
#
# import math
# from functools import reduce
#
# import numpy as np
# import torch
# from ogb.utils import smiles2graph
# from rdkit import Chem
# from rdkit.Chem import BRICS, Recap
# from torch.distributions.normal import Normal
# from torch_geometric.data import Data
#
# from GOOD import register
# from GOOD.utils.config_reader import Union, CommonArgs, Munch
# from .BaseGNN import GNNBasic
# from .Classifiers import Classifier
# from .GINs import GINFeatExtractor
# from .GINvirtualnode import vGINFeatExtractor
#
#
# @register.model_register
# class MRLGIN(GNNBasic):
#
#     def __init__(self, config: Union[CommonArgs, Munch], backend_class=GINFeatExtractor):
#         super(MRLGIN, self).__init__(config)
#         self.domain_classifier = DomainClassifier(
#             backend_dim=config.model.dim_hidden,
#             backend=backend_class(config), num_domains=config.dataset.num_envs,
#             num_tasks=config.dataset.num_classes
#         )
#         self.conditional_gnn = ConditionalGnn(
#             emb_dim=config.model.dim_hidden,
#             backend_dim=config.model.dim_hidden,
#             backend=backend_class(config), num_classes=config.dataset.num_classes,
#             num_domain=config.dataset.num_envs
#         )
#         self.main_model = Framework(
#             base_model=backend_class(config), sub_model=backend_class(config),
#             base_dim=config.model.dim_hidden,
#             sub_dim=config.model.dim_hidden,
#             num_tasks=config.dataset.num_classes, dropout=config.model.dropout_rate
#         )
#
#         self.classifier = Classifier(config)
#         self.learn_edge_att = config.ood.extra_param[0]
#         self.config = config
#
#     def forward(self, *args, **kwargs):
#         r"""
#         The GSAT model implementation.
#
#         Args:
#             *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
#             **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
#
#         Returns (Tensor):
#             Label predictions and other results for loss calculations.
#
#         """
#         if self.at_stage(1):
#             print(f'[INFO] training on assistant models on {ep} epoch')
#             self.conditional_gnn.train()
#             self.domain_classifier.train()
#             Eqs, ELs = [], []
#             for subs, graphs in tqdm(train_loader):
#                 # subs: a list of string, eval(string) can
#                 # get the substructures of corresponding molecule
#                 graphs = graphs.to(device)
#                 q_e = torch.softmax(domain_classifier(graphs), dim=-1)
#                 losses, batch_size = [], len(subs)
#
#                 for dom in range(args.num_domain):
#                     domain_info = torch.ones(batch_size).long().to(device)
#                     p_ye = conditional_gnn(graphs, domain_info * dom)
#
#                     labeled = graphs.y == graphs.y
#                     # there are nan in the labels so use this to mask them
#                     # and this is a multitask binary classification
#                     data_belong = torch.arange(batch_size).long()
#                     data_belong = data_belong.unsqueeze(dim=-1).to(device)
#                     data_belong = data_belong.repeat(1, dataset.num_tasks)
#                     # [batch_size, num_tasks] same as p_ye
#
#                     loss = bce_log(p_ye[labeled], graphs.y[labeled].float())
#                     # shape: [numbers of not nan gts]
#                     batch_loss = torch_scatter.scatter(
#                         loss, dim=0, index=data_belong[labeled],
#                         reduce='mean'
#                     )  # [batch_size]
#                     # considering the dataset is a multitask binary
#                     # classification task, the process above is to
#                     # get a average loss among all the tasks,
#                     # when there is only one task, it's equilvant to
#                     # bce_with_logit without reduction
#                     losses.append(batch_loss)
#
#                 losses = torch.stack(losses, dim=1)  # [batch_size, num_domain]
#                 Eq = torch.mean(torch.sum(q_e * losses, dim=-1))
#                 ELBO = Eq + KLDist(q_e, prior)
#
#                 ELBO.backward()
#                 optimizer_con.step()
#                 optimizer_dom.step()
#                 optimizer_dom.zero_grad()
#                 optimizer_con.zero_grad()
#
#                 Eqs.append(Eq.item())
#                 ELs.append(ELBO.item())
#             print(f'[INFO] Eq: {np.mean(Eqs)}, ELBO: {np.mean(ELs)}')
#             if best_ep is None or np.mean(ELs) < min_loss:
#                 min_loss, best_ep = np.mean(ELs), ep
#                 best_para = {
#                     'con': deepcopy(conditional_gnn.state_dict()),
#                     'dom': deepcopy(domain_classifier.state_dict())
#                 }
#
#     def sampling(self, att_log_logits, training):
#         att = self.concrete_sample(att_log_logits, temp=1, training=training)
#         return att
#
#     @staticmethod
#     def lift_node_att_to_edge_att(node_att, edge_index):
#         src_lifted_att = node_att[edge_index[0]]
#         dst_lifted_att = node_att[edge_index[1]]
#         edge_att = src_lifted_att * dst_lifted_att
#         return edge_att
#
#     @staticmethod
#     def concrete_sample(att_log_logit, temp, training):
#         if training:
#             random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
#             random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
#             att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
#         else:
#             att_bern = (att_log_logit).sigmoid()
#         return att_bern
#
#
# @register.model_register
# class MRLvGIN(MRLGIN):
#     r"""
#     The GIN virtual node version of GSAT.
#     """
#
#     def __init__(self, config: Union[CommonArgs, Munch]):
#         super(MRLvGIN, self).__init__(config)
#         self.gnn = vGINFeatExtractor(config)
#
#
# class AttentionAgger(torch.nn.Module):
#     def __init__(self, Qdim, Kdim, Mdim):
#         super(AttentionAgger, self).__init__()
#         self.model_dim = Mdim
#         self.WQ = torch.nn.Linear(Qdim, Mdim)
#         self.WK = torch.nn.Linear(Qdim, Mdim)
#
#     def forward(self, Q, K, V, mask=None):
#         Q, K = self.WQ(Q), self.WK(K)
#         Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
#         if mask is not None:
#             Attn = torch.masked_fill(Attn, mask, -(1 << 32))
#         Attn = torch.softmax(Attn, dim=-1)
#         return torch.matmul(Attn, V)
#
#
# class Framework(torch.nn.Module):
#     def __init__(
#             self, base_model, sub_model, num_tasks,
#             base_dim, sub_dim, dropout=0.5,
#     ):
#         super(Framework, self).__init__()
#         self.base_model = base_model
#         self.sub_model = sub_model
#         self.attenaggr = AttentionAgger(
#             base_dim, sub_dim, max(base_dim, sub_dim)
#         )
#         predictor_layers = [torch.nn.Linear(sub_dim, sub_dim)]
#         if dropout < 1 and dropout > 0:
#             predictor_layers.append(torch.nn.Dropout(dropout))
#         predictor_layers.append(torch.nn.ReLU())
#         predictor_layers.append(torch.nn.Linear(sub_dim, num_tasks))
#         self.predictor = torch.nn.Sequential(*predictor_layers)
#
#     def sub_feature_from_graphs(self, subs, device, return_mask=False):
#         substructure_graph, mask = graph_from_substructure(subs, True, 'pyg')
#         substructure_graph = substructure_graph.to(device)
#         substructure_feat = self.sub_model(substructure_graph)
#         return (substructure_feat, mask) if return_mask else substructure_feat
#
#     def forward(self, substructures, batched_data):
#         graph_feat = self.base_model(batched_data)
#         substructure_feat, mask = self.sub_feature_from_graphs(
#             subs=substructures, device=batched_data.x.device,
#             return_mask=True
#         )
#         mask = torch.from_numpy(mask).to(batched_data.x.device)
#         Attn_mask = torch.logical_not(mask)
#         molecule_feat = self.attenaggr(
#             Q=graph_feat, K=substructure_feat,
#             V=substructure_feat, mask=Attn_mask
#         )
#         result = self.predictor(molecule_feat)
#         return result
#
#
# class MeanLoss(torch.nn.Module):
#     def __init__(self, base_loss):
#         super(MeanLoss, self).__init__()
#         self.base_loss = base_loss
#
#     def forward(self, pred, gt, domain):
#         _, group_indices, _ = split_into_groups(domain)
#         total_loss, total_cnt = 0, 0
#         for i_group in group_indices:
#             pred_group, gt_group = pred[i_group], gt[i_group]
#             islabeled = gt_group == gt_group
#             total_loss += self.base_loss(
#                 pred_group[islabeled], gt_group[islabeled]
#             )
#             total_cnt += 1
#         return total_loss / total_cnt
#
#
# class DeviationLoss(torch.nn.Module):
#     def __init__(self, activation, reduction='mean'):
#         super(DeviationLoss, self).__init__()
#         assert activation in ['relu', 'abs', 'none'], \
#             'Invaild activation function'
#         assert reduction in ['mean', 'sum'], \
#             'Invalid reduction method'
#
#         self.activation = activation
#         self.reduction = reduction
#
#     def forward(self, pred, condition_pred_mean):
#         if self.activation == 'relu':
#             loss = torch.relu(pred - condition_pred_mean)
#         elif self.activation == 'abs':
#             loss = torch.abs(pred - condition_pred_mean)
#         else:
#             loss = pred - condition_pred_mean
#
#         if self.reduction == 'mean':
#             return torch.mean(loss)
#         else:
#             return torch.sum(loss)
#
#
# class ConditionalGnn(torch.nn.Module):
#     def __init__(self, emb_dim, backend_dim, backend, num_domain, num_classes):
#         super(ConditionalGnn, self).__init__()
#         self.emb_dim = emb_dim
#         self.class_emb = torch.nn.Parameter(
#             torch.zeros(num_domain, emb_dim)
#         )
#         self.backend = backend
#         self.predictor = torch.nn.Linear(backend_dim + emb_dim, num_classes)
#
#     def forward(self, batched_data, domains):
#         domain_feat = torch.index_select(self.class_emb, 0, domains)
#         graph_feat = self.backend(batched_data)
#         result = self.predictor(torch.cat([graph_feat, domain_feat], dim=1))
#         return result
#
#
# class DomainClassifier(torch.nn.Module):
#     def __init__(self, backend_dim, backend, num_domains, num_tasks):
#         super(DomainClassifier, self).__init__()
#         self.backend = backend
#         self.predictor = torch.nn.Linear(backend_dim + num_tasks, num_domains)
#
#     def forward(self, batched_data):
#         graph_feat = self.backend(batched_data)
#         y_part = torch.nan_to_num(batched_data.y).float()
#         return self.predictor(torch.cat([graph_feat, y_part], dim=-1))
#
#
# def KLDist(p, q, eps=1e-8):
#     log_p, log_q = torch.log(p + eps), torch.log(q + eps)
#     return torch.sum(p * (log_p - log_q))
#
#
# def bce_log(pred, gt, eps=1e-8):
#     prob = torch.sigmoid(pred)
#     return -(gt * torch.log(prob + eps) + (1 - gt) * torch.log(1 - prob + eps))
#
#
# def discrete_gaussian(nums, std=1):
#     Dist = Normal(loc=0, scale=1)
#     plen, halflen = std * 6 / nums, std * 3 / nums
#     posx = torch.arange(-3 * std + halflen, 3 * std, plen)
#     result = Dist.cdf(posx + halflen) - Dist.cdf(posx - halflen)
#     return result / result.sum()
#
#
# def split_into_groups(g):
#     unique_groups, unique_counts = torch.unique(
#         g, sorted=False, return_counts=True
#     )
#     group_indices = [
#         torch.nonzero(g == group, as_tuple=True)[0]
#         for group in unique_groups
#     ]
#     return unique_groups, group_indices, unique_counts
#
#
# def get_substructure(mol=None, smile=None, decomp='brics'):
#     assert mol is not None or smile is not None, \
#         'need at least one info of mol'
#     assert decomp in ['brics', 'recap'], 'Invalid decomposition method'
#     if mol is None:
#         mol = Chem.MolFromSmiles(smile)
#
#     if decomp == 'brics':
#         substructures = BRICS.BRICSDecompose(mol)
#     else:
#         recap_tree = Recap.RecapDecompose(mol)
#         leaves = recap_tree.GetLeaves()
#         substructures = set(leaves.keys())
#     return substructures
#
#
# def substructure_batch(smiles, return_mask, return_type='numpy'):
#     sub_structures = [get_substructure(smile=x) for x in smiles]
#     return graph_from_substructure(sub_structures, return_mask, return_type)
#
#
# def graph_from_substructure(subs, return_mask=False, return_type='numpy'):
#     sub_struct_list = list(reduce(lambda x, y: x.update(y) or x, subs, set()))
#     sub_to_idx = {x: idx for idx, x in enumerate(sub_struct_list)}
#     mask = np.zeros([len(subs), len(sub_struct_list)], dtype=bool)
#     sub_graph = [smiles2graph(x) for x in sub_struct_list]
#     for idx, sub in enumerate(subs):
#         mask[idx][list(sub_to_idx[t] for t in sub)] = True
#
#     edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
#     for idx, graph in enumerate(sub_graph):
#         edge_idxes.append(graph['edge_index'] + lstnode)
#         edge_feats.append(graph['edge_feat'])
#         node_feats.append(graph['node_feat'])
#         lstnode += graph['num_nodes']
#         batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)
#
#     result = {
#         'edge_index': np.concatenate(edge_idxes, axis=-1),
#         # 'node_feat': np.concatenate(node_feats, axis=0),
#         'edge_attr': np.concatenate(edge_feats, axis=0),
#         'batch': np.concatenate(batch, axis=0),
#         'x': np.concatenate(node_feats, axis=0)
#     }
#
#     assert return_type in ['numpy', 'torch', 'pyg'], 'Invaild return type'
#     if return_type in ['torch', 'pyg']:
#         for k, v in result.items():
#             result[k] = torch.from_numpy(v)
#
#     result['num_nodes'] = lstnode
#
#     if return_type == 'pyg':
#         result = Data(**result)
#
#     if return_mask:
#         return result, mask
#     else:
#         return result

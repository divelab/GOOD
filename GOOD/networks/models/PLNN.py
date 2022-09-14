import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import Tensor
from torch.nn.functional import gumbel_softmax
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .ICBaseGNN import GNNBasic
from .PLGINs import GINFeatExtractor
from .PLGINvirtualnode import vGINFeatExtractor
from typing import Tuple
from GOOD.utils.train import at_stage, gumbel_sigmoid


class classifier(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""
    def __init__(self, config: Union[CommonArgs, Munch], temperature=1.0):
        super(classifier, self).__init__()
        num_classes = config.dataset.num_classes if config.dataset.num_classes != 1 else 2
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, config.model.dim_hidden))
        self.distance_scale = nn.Parameter(torch.Tensor(1))
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        distances = torch.abs(self.distance_scale) * torch.cdist(F.normalize(features), F.normalize(self.prototypes), p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = - distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature

@register.model_register
class PL_GIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feature_extractor = GINFeatExtractor(config)

        # self.classifier = nn.Sequential(*(
        #     [nn.Linear(config.model.dim_hidden, config.dataset.num_classes)]
        # ))
        self.classifier = classifier(config)

        self.config = config
        self.threshold = 0.6
        self.ratio = config.ood.ood_param
        self.max_k = config.ood.extra_param[0]

        # --- special control
        self.eps = 1e-8
        self.filter_non_matched = False
        self.euclidean = True
        self.stochastic = True
        self.soft_pick = False

    def forward(self, *args, **kwargs) -> Tuple:
        if self.training:
            paired_data = kwargs.get('data')
            batch = paired_data.batch
            num_pair = torch.div((batch.max() + 1), 2, rounding_mode='floor')
            out_readout = self.feature_extractor(*args, **kwargs)

            matched_out = []
            matching_rate = []
            for pair_id in range(num_pair):
                out_a = out_readout[batch == pair_id]
                out_b = out_readout[batch == (pair_id + num_pair)]
                if out_a.shape[0] > out_b.shape[0]:
                    out_a, out_b = out_b, out_a

                # --- calculate distance ---
                if self.filter_non_matched:
                    cos_sim, match_map = self.sim(out_a, out_b)
                else:
                    cos_sim = self.sim(out_a, out_b)

                # --- choose the best matched features' indices ---
                if self.stochastic:
                    if self.soft_pick:
                        sim_a = gumbel_softmax(cos_sim, dim=1)
                        max_a = cos_sim.max(1)[0]
                        noise_max_a = gumbel_softmax(max_a, dim=0)
                        soft_pick = sim_a * noise_max_a[:, None]
                        # --- Matched pooling ---
                        if self.config.model.global_pool == 'mean':
                            if self.filter_non_matched:
                                matched_out.append(
                                    ((out_a[pick_a] + out_b[pick_b]) * match_map[pick_a, pick_b]).sum(0) / (
                                            match_map[pick_a, pick_b].sum(0) + self.eps))
                            else:
                                matched_out.append(((out_a[:, None, :] + out_b[None, :, :]) / 2 * soft_pick[..., None]).sum(0).sum(0))
                                # matched_out.append((out_a[pick_a] + out_b[pick_b]).mean(0))
                        else:
                            matched_out.append((out_a[pick_a] + out_b[pick_b]).max(0)[0])
                    else:
                        sim_a = gumbel_softmax(cos_sim, dim=1)
                        max_a = cos_sim.max(1)[0]
                        noise_max_a = gumbel_softmax(max_a, dim=0)
                        k = min(math.ceil(cos_sim.shape[0] * self.ratio), self.max_k)
                        pick_a = torch.topk(noise_max_a, k, dim=0).indices
                        prob_pick_b = sim_a[pick_a]
                        pick_b = prob_pick_b.argmax(1)
                        # sim_b = cos_sim.max(0)[0]
                        # sim_max = cos_sim.max()
                        # sim_min = cos_sim.min()
                        # norm_sim_a = (sim_a - sim_min) / (sim_max - sim_min + self.eps)
                        # norm_sim_b = (sim_b - sim_min) / (sim_max - sim_min + self.eps)
                        # sampled_a = gumbel_sigmoid(norm_sim_a)
                        # sampled_b = gumbel_sigmoid(norm_sim_b)

                        # --- Matched pooling ---
                        if self.config.model.global_pool == 'mean':
                            if self.filter_non_matched:
                                matched_out.append(((out_a[pick_a] + out_b[pick_b]) * match_map[pick_a, pick_b]).sum(0) / (
                                            match_map[pick_a, pick_b].sum(0) + self.eps))
                            else:
                                matched_out.append((out_a[pick_a] + out_b[pick_b]).mean(0))
                        else:
                            matched_out.append((out_a[pick_a] + out_b[pick_b]).max(0)[0])

                else:

                    max_a, pick_b = cos_sim.max(1)
                    k = min(math.ceil(cos_sim.shape[0] * self.ratio), self.max_k)
                    pick_a = torch.topk(max_a, k, dim=0).indices
                    # if pick_a.sum() == 0:
                    #     pick_a = max_a.argmax().unsqueeze(0)
                    pick_b = pick_b[pick_a]
                    matching_rate.append(max_a[pick_a].mean())

                    # --- Matched pooling ---
                    if self.config.model.global_pool == 'mean':
                        if self.filter_non_matched:
                            matched_out.append(((out_a[pick_a] + out_b[pick_b]) * match_map[pick_a, pick_b]).sum(0) / (match_map[pick_a, pick_b].sum(0) + self.eps))
                        else:
                            matched_out.append((out_a[pick_a] + out_b[pick_b]).mean(0))
                    else:
                        matched_out.append((out_a[pick_a] + out_b[pick_b]).max(0)[0])

            # print(f'Max cosine similarity: {max_a.max()}')

            matched_out = torch.stack(matched_out, 0)
            out = self.classifier(matched_out)
        else:
            data = kwargs.get('data')
            batch = data.batch
            out_readout = self.feature_extractor(*args, **kwargs)
            matched_out = []
            for i in range(batch[-1] + 1):
                out_a = out_readout[batch == i]

                # --- calculate distance ---

                if self.filter_non_matched:
                    cos_sim, match_map = self.sim(out_a, self.classifier.prototypes.data)
                else:
                    cos_sim = self.sim(out_a, self.classifier.prototypes.data)

                if self.stochastic and False:
                    sim_a = cos_sim.max(1)[0]
                    sim_max = cos_sim.max()
                    sim_min = cos_sim.min()
                    norm_sim_a = (sim_a - sim_min) / (sim_max - sim_min)
                    sampled_a = gumbel_sigmoid(norm_sim_a)

                    # --- Matched pooling ---
                    if self.config.model.global_pool == 'mean':
                        matched_out.append((out_a * sampled_a[:, None]).sum(0) / (sampled_a.sum() + self.eps))
                    else:
                        raise ValueError('Not support pooling except mean.')
                else:

                    k = min(math.ceil(cos_sim.shape[0] * self.ratio), self.max_k)
                    top_cs, pick = torch.topk(cos_sim, k, dim=0)
                    selected_class = top_cs.sum(0).argmax()
                    pick_a = pick[:, selected_class]
                    assert not torch.isnan(out_a[pick_a].mean(0)).sum()
                    if self.config.model.global_pool == 'mean':
                        if self.filter_non_matched:
                            matched_out.append((out_a[pick_a] * match_map[pick_a, selected_class]).sum(0) / (match_map[pick_a, selected_class].sum(0) + self.eps))
                        else:
                            matched_out.append(out_a[pick_a].mean(0))
                    else:
                        matched_out.append(out_a[pick_a].max(0)[0])
            matched_out = torch.stack(matched_out, 0)
            # matching_rate = torch.stack(matching_rate, 0)
            # num_classes = self.config.dataset.num_classes if self.config.dataset.num_classes != 1 else 2
            # matching_rate = matching_rate.reshape(num_classes, -1)
            # matching_label = matching_rate.argmax(0)
            # eye = torch.eye(num_classes)
            # idx = eye[:, matching_label].bool()
            # matched_out = matched_out.reshape(num_classes, -1, self.config.model.dim_hidden).transpose(0, 1)[idx.T]

            out = self.classifier(matched_out)
        assert not torch.isnan(out).sum()
        return out

    def sim(self, a, b, eps=1e-8):
        mod_a = a.norm(dim=1).unsqueeze(1)
        mod_b = b.norm(dim=1).unsqueeze(1)
        assert not (mod_a < eps).sum()
        norm_a = a / torch.clamp(mod_a, min=eps)
        norm_b = b / torch.clamp(mod_b, min=eps)
        if self.euclidean:
            if self.filter_non_matched:
                diff_matrix = norm_a[:, None, :] - norm_b[None, ...]
                match_filter = diff_matrix.abs() < 0.2
                sim = - ((diff_matrix ** 2) * match_filter).sum(-1).sqrt()
                return sim, match_filter
            else:
                diff_matrix = norm_a[:, None, :] - norm_b[None, ...]
                sim = - (diff_matrix ** 2).sum(-1).sqrt()
                assert not torch.isnan(sim).sum()
                return sim
        else:
            if self.filter_non_matched:
                dot_production_matrix = norm_a[:, None, :] * norm_b[None, ...]
                positive_value_filter = dot_production_matrix > 0
                cos_sim_matrix = (positive_value_filter * dot_production_matrix).sum(-1)
                return cos_sim_matrix, positive_value_filter
            else:
                cos_sim_matrix = norm_a @ norm_b.T
                assert not torch.isnan(cos_sim_matrix).sum()
                return cos_sim_matrix

@register.model_register
class PL_vGIN(PL_GIN):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feature_extractor = vGINFeatExtractor(config)


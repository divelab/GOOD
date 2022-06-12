import torch
import torch.nn as nn
from torch.autograd import Function

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GCNs import GCNFeatExtractor


@register.model_register
class DANN_GCN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = GCNFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

        self.dc = nn.Linear(config.model.dim_hidden, config.dataset.num_envs)
        self.config = config

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :return:
        """
        out_readout = self.feat_encoder(*args, **kwargs)

        dc_out = GradientReverseLayerF.apply(out_readout, self.config.train.alpha)
        dc_out = self.dc(dc_out)

        out = self.classifier(out_readout)
        return out, dc_out


class GradientReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)  # * alpha

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

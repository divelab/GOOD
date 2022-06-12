import torch
import torch.nn as nn

from GOOD.utils.config_reader import Union, CommonArgs, Munch


class Classifier(torch.nn.Module):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Classifier, self).__init__()
        # self.classifier = nn.Sequential(*(
        #         [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_ffn), nn.BatchNorm1d(2 * config.model.dim_ffn)] +
        #         [nn.ReLU(), nn.Linear(2 * config.model.dim_ffn, config.dataset.num_classes)]
        # ))
        self.classifier = nn.Sequential(*(
            [nn.Linear(config.model.dim_hidden, config.dataset.num_classes)]
        ))

    def forward(self, feat):
        return self.classifier(feat)

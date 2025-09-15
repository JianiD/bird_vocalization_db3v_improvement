from collections import OrderedDict

import torch
from torch import nn

from .layers import DenseLayer, StatsPool, TDNNLayer, get_nonlinear, LSTM, my_transpose

class RelaxedIFN(nn.Module):
    def __init__(self, num_features):
        super(RelaxedIFN, self).__init__()
        self.eps = 1e-5
        self.gamma = nn.Parameter(torch.ones(num_features, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + self.eps
        norm_x = (x - mean) / std
        gate = self.sigmoid(self.gamma)
        x = gate * norm_x + (1 - gate) * x
        return x

class TDNN(nn.Module):
    def __init__(self,
                 feat_dim=30,
                 embedding_size=512,
                 num_classes=None,
                 config_str='relu',
                 config_str2='batchnorm-relu'):
        super(TDNN, self).__init__()

        self.xvector = nn.Sequential(OrderedDict([
            ('relaxed_ifn', RelaxedIFN(feat_dim)),
            ('tdnn1', TDNNLayer(feat_dim, 512, 5, dilation=1, padding=-1, config_str=config_str)),
            ('tdnn2', TDNNLayer(512, 512, 3, dilation=2, padding=-1, config_str=config_str)),
            ('tdnn3', TDNNLayer(512, 512, 3, dilation=3, padding=-1, config_str=config_str)),
            ('tdnn4', DenseLayer(512, 512, config_str=config_str)),
            ('tdnn5', DenseLayer(512, 1500, config_str=config_str)),
            ('stats', StatsPool()),
            ('affine', nn.Linear(3000, embedding_size))
        ]))

        self.nonlinear = get_nonlinear(config_str2, embedding_size)
        self.dense = DenseLayer(embedding_size, embedding_size, config_str=config_str2)

        if num_classes is not None:
            self.classifier = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        xvector = self.xvector(x)
        x = self.dense(self.nonlinear(xvector))
        x = self.classifier(x)
        return x, xvector

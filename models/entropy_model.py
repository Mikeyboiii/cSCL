import torch
from torch import nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


class MeanScaleHyperPrior(nn.Module):
    '''
    A hyper prior entropy model for estimating the entropy of a representation
    '''
    def __init__(self, z_dim):
        super().__init__()
    

    def forward(self, z):


import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from .entropy_model import HyperPrior, FactorizedPrior

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Projection(nn.Module):
    def __init__(self, c_in=2048, c_out=256):
        super().__init__()
        self.c_in = c_in
        self.c_h = c_in
        self.c_out = c_out
        
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.c_in, self.c_h, bias=True),
            nn.BatchNorm1d(self.c_h),
            nn.ReLU(),
            nn.Linear(self.c_h, self.c_out, bias=False)
        )
    def forward(self, x):
        x = self.layers(x)
        return F.normalize(x, dim=1)

class simclr(nn.Module):
    def __init__(self, z_dim=256, arch='resnet50', entropy_model='factorized'):
        super().__init__()
        assert arch in ['resnet50', 'resnet18']
        assert entropy_model in ['factorized', 'hyperprior', None]
        if arch == 'resnet50':
            backbone = resnet50(pretrained=False)
            c_in = 2048
        elif arch == 'resnet18':
            backbone = resnet18(pretrained=False)
            c_in = 512
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.projector = Projection(c_in=c_in, c_out=z_dim)

        if entropy_model == 'factorized':
            self.entropy_model = FactorizedPrior(z_dim)
        elif entropy_model == 'hyperprior':
            self.entropy_model = HyperPrior(z_dim, 64)
        else:
            self.entropy_model = None


    def forward(self, x):
        
        h = self.encoder(x)
        h_hat = Quantize.apply(h)
        z = self.projector(h_hat)

        rate = torch.tensor([0]).to(device)

        if self.entropy_model is not None:
            rate = self.entropy_model(h)

        return z, rate
        

class c_resnet(nn.Module):
    def __init__(self, num_classes=10, arch='resnet50', compress=False, entropy_model='factorized'):
        super().__init__()
        assert arch in ['resnet50', 'resnet18']
        assert entropy_model in ['factorized', 'hyperprior', None]
        if arch == 'resnet50':
            backbone = resnet50(pretrained=False)
            c_in = 2048
        elif arch == 'resnet18':
            backbone = resnet18(pretrained=False)
            c_in = 512
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.fc = nn.Linear(c_in, num_classes)

        if entropy_model == 'factorized':
            self.entropy_model = FactorizedPrior(c_in)
        elif entropy_model == 'hyperprior':
            self.entropy_model = HyperPrior(c_in, 64)
        else:
            self.entropy_model = None


    def forward(self, x):
        
        h = self.encoder(x)
        h_hat = Quantize.apply(h)
        out = torch.flatten(h_hat, 1)
        out = self.fc(out)

        rate = torch.tensor([0]).to(device)

        if self.entropy_model is not None:
            rate = self.entropy_model(h)

        return out, rate


if __name__ == '__main__':
    #x = torch.rand([4, 3, 224, 224])
    #y = torch.rand([4, 3, 224, 224])
    #model = simclr(arch='resnet18')
    #zx, zy = model(torch.cat([x, y], dim=0)).chunk(2, dim=0)
    #print(zx.shape, zy.shape)


    print(encoder(x).shape)
    #loss = nt_xent_loss(zx, zy, 0.1, eps=1e-6)

    #print(loss)


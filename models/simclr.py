import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from .entropy_model import HyperPrior, FactorizedPrior, conv, deconv

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
            self.entropy_model = HyperPrior(c_in, c_in * 2)
        else:
            self.entropy_model = None

        self.round = Quantize.apply


    def forward(self, x):
        
        h = self.encoder(x)
        h_hat  = self.round(h)

        out = torch.flatten(h_hat, 1)
        out = self.fc(out)

        rate = torch.tensor([0]).to(device)

        if self.entropy_model is not None:
            rate = self.entropy_model(h_hat)

        return out, rate


class c_CNN(nn.Module):
    def __init__(self, N=64, M=96, num_classes=10):
        super().__init__()
        #assert entropy_model in ['factorized', 'hyperprior', None]

        self.encoder = nn.Sequential(
            conv(3, N),
            nn.BatchNorm2d(N),
            nn.ReLU(),
            conv(N, N),
            nn.BatchNorm2d(N),
            nn.ReLU(),
            conv(N, N),
            #nn.BatchNorm2d(N),
            #nn.ReLU(),
            #conv(N, N),
        )

        self.hyper_encoder = nn.Sequential(
            conv(N, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M, M),
            nn.LeakyReLU(inplace=True),
            conv(M, M),
        )

        self.hyper_decoder = nn.Sequential(
            deconv(M, N),
            nn.LeakyReLU(inplace=True),
            deconv(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(N * 3 // 2, N * 2, stride=1, kernel_size=3),
        )


        self.factorized_entropy = EntropyBottleneck(M)
        self.gaussian_conditional = GaussianConditional(None)

        self.round = Quantize.apply
        self.fc = nn.Linear(1024, num_classes)


    def forward(self, x):

        y = self.encoder(x)
        z = self.hyper_encoder(y)
        y_num = y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3]
        z_num = z.shape[0] * z.shape[1] * z.shape[2] * z.shape[3]


        z_hat, z_llh = self.factorized_entropy(z)
        gaussian_params = self.hyper_decoder(z_hat)

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_llh = self.gaussian_conditional(y, scales_hat, means=means_hat)

        y_hat  = self.round(y)

        out = torch.flatten(y_hat, 1)
        out = self.fc(out)

        rate_y =  (torch.sum(-1.0*torch.log2(y_llh)) / y_num)
        rate_z =  (torch.sum(-1.0*torch.log2(z_llh)) / z_num)

        #rate = torch.tensor([0]).to(device)

        #if self.entropy_model is not None:
        #    rate = self.entropy_model(h_hat)

        return out, rate_y + rate_z



if __name__ == '__main__':
    #x = torch.rand([4, 3, 224, 224])
    #y = torch.rand([4, 3, 224, 224])
    #model = simclr(arch='resnet18')
    #zx, zy = model(torch.cat([x, y], dim=0)).chunk(2, dim=0)
    #print(zx.shape, zy.shape)


    print(encoder(x).shape)
    #loss = nt_xent_loss(zx, zy, 0.1, eps=1e-6)

    #print(loss)


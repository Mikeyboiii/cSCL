import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from .entropy_model import HyperPrior, FactorizedPrior, conv, deconv, SmallHyperPrior

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
    def __init__(self, z_dim=256, arch='resnet50', entropy_model='hyperprior'):
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
            self.entropy_model = FactorizedPrior(c_in)
        elif entropy_model == 'hyperprior':
            self.entropy_model = HyperPrior(c_in, c_in * 2)
        else:
            self.entropy_model = None

        self.round = Quantize.apply


    def forward(self, x):
        
        h = self.encoder(x)
        h_hat = self.round(h)
        z = self.projector(h_hat)

        rate = torch.tensor([0]).to(device)

        if self.entropy_model is not None:
            rate = self.entropy_model(h)

        return z, rate
        

class c_resnet(nn.Module):
    def __init__(self, num_classes=10, arch='resnet50', entropy_model='hyperprior'):
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


class c_resnet_mid(nn.Module):
    def __init__(self, num_classes=10, arch='resnet50', entropy_model='hyperprior'):
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
        )
        self.decoder = nn.Sequential(
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

        self.h_e = nn.Sequential(
            conv(64, 96, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(96, 96),
            nn.LeakyReLU(inplace=True),
            conv(96, 96),
        )

        self.h_d = nn.Sequential(
            deconv(96, 64),
            nn.LeakyReLU(inplace=True),
            deconv(64, 96),
            nn.LeakyReLU(inplace=True),
            deconv(96, 128, stride=1, kernel_size=3),
        )


        self.fc = nn.Linear(c_in, num_classes)
        self.round = Quantize.apply

        self.factorized = EntropyBottleneck(96)
        self.gaussian_conditional = GaussianConditional(None)


    def forward(self, x):
        
        h = self.encoder(x)
        z = self.h_e(h)

        h_num = h.shape[0] * h.shape[1] * h.shape[2] * h.shape[3]
        z_num = z.shape[0] * z.shape[1] * z.shape[2] * z.shape[3]

        z_hat, z_llh = self.factorized(z)
        gaussian_params = self.h_d(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, h_llh = self.gaussian_conditional(h, scales_hat, means=means_hat)
        
        h_hat = self.round(h)
        out = self.decoder(h_hat)
        out = torch.flatten(out, 1)

        out = self.fc(out)

        rate_h =  (torch.sum(-1.0*torch.log2(h_llh)) / h_num)
        rate_z =  (torch.sum(-1.0*torch.log2(z_llh)) / z_num)

        return out, rate_h + rate_z, h_hat


class MLP(nn.Module):
    def __init__(self, ent_model='hyperprior'):
        super().__init__()

        #self.entropy_model = FactorizedPrior(256)
        if ent_model == 'hyperprior':
            self.entropy_model =  SmallHyperPrior(256, 256)#SmallHyperPrior(256,  256)
        elif ent_model == 'factorized':
            self.entropy_model =  FactorizedPrior(256)
        elif ent_model == None:
            self.entropy_model = None

        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
        )
        self.fc = nn.Linear(256, 10)
        self.round = Quantize.apply

        self.apply(self._init_weights)
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)
        #h_hat = h
        h_hat  = self.round(h)
        #print(h_hat.min().item(), h_hat.max().item())
        out = self.fc(h_hat)

        rate = torch.tensor([0]).to(device)

        if self.entropy_model is not None:
            rate = self.entropy_model(h_hat.unsqueeze(2).unsqueeze(3))

        return out, rate



class AE_fc(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
        )
        self.entropy_model = FactorizedPrior(256)
        self.round = Quantize.apply

    def forward(sellf, x):
        y = self.encoder(x)
        y_hat  = self.round(y)
        x_hat = self.decoder(y_hat)

        rate = self.entropy_model(y_hat))

        return x_hat, rate


if __name__ == '__main__':
    #x = torch.rand([4, 3, 224, 224])
    #y = torch.rand([4, 3, 224, 224])
    #model = simclr(arch='resnet18')
    #zx, zy = model(torch.cat([x, y], dim=0)).chunk(2, dim=0)
    print(zx.shape, zy.shape)


    #model = c_resnet(num_classes=10, arch='resnet18', entropy_model='hyperprior')
    #print(model.encoder[7][1].conv2)
    #loss = nt_xent_loss(zx, zy, 0.1, eps=1e-6)

    #print(loss)


import torch
from torch import nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import math
from torchvision.models import resnet18
#from .entropy.bottleneck import EntropyBottleneck, Gaussian_Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def conv(in_channels, out_channels, kernel_size=3, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=3, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class Compressor(nn.Module):
    '''
    A Mean Scale hyper prior entropy model for estimating the entropy of a representation
    '''
    def __init__(self, N, M):
        super().__init__()
        self.hyper_encoder = nn.Sequential(
            nn.Linear(N, M),
            nn.LeakyReLU(inplace=True),
            nn.Linear(M, M),
            nn.LeakyReLU(inplace=True),
            nn.Linear(M, M),
        )
        self.hyper_decoder = nn.Sequential(
            nn.Linear(M, M),
            nn.LeakyReLU(inplace=True),
            nn.Linear(M, M),
            nn.LeakyReLU(inplace=True),
            nn.Linear(M, N),
            nn.LeakyReLU(inplace=True),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(N, N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N, N),
        )
        self.std_head = nn.Sequential(
            nn.Linear(N, N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N, N),
        )

        self.factorized_entropy = EntropyBottleneck(M)
        self.gaussian_entropy = Gaussian_Model()
        

    def forward(self, y):
        y = y.unsqueeze(2).unsqueeze(2)
        noise = torch.rand_like(y) - 0.5
        y_hat = y + noise.to(device)

        z = self.hyper_encoder(y.squeeze(2).squeeze(2))
 
        z_hat, z_bits = self.factorized_entropy(z.unsqueeze(2).unsqueeze(3))

        hyper_out = self.hyper_decoder(z_hat.squeeze(2).squeeze(2))


        mu = self.mean_head(hyper_out).unsqueeze(2).unsqueeze(2)
        sigma = self.std_head(hyper_out).unsqueeze(2).unsqueeze(2)


        y_bits = self.gaussian_entropy(y, sigma, mu)

        rate_y, rate_z = y_bits/y.shape[1] , z_bits/z.shape[1]

        print(rate_y, rate_z)

        return y_hat.squeeze(0).squeeze(0), rate_y, rate_z


class FactorizedCompressor(nn.Module):
    '''
    A Factorized entropy model for estimating the entropy of a representation
    '''
    def __init__(self, z_dim):
        super().__init__()
        self.factorized_entropy = EntropyBottleneck(z_dim)
        
    def forward(self, z):
        
        z_hat, z_likelihoods = self.factorized_entropy(z)
        #print(z, z_likelihoods)
        #print(z_likelihoods)

        bpp = torch.sum(-1.0*torch.log2(z_likelihoods)) / (z.shape[0] * z.shape[1] * z.shape[2] * z.shape[3])

        return bpp

class AECompress(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.factorized_entropy = EntropyBottleneck(M)
        backbone = resnet18(pretrained=False)
        self.g_a = nn.Sequential(
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

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.ReLU(),
            deconv(N, N),
            nn.ReLU(),
            deconv(N, N),
            nn.ReLU(),
            deconv(N, N),
            nn.ReLU(),
            deconv(N, 3),
        )

        self.N = N
        self.M = M

    def forward(self, x):
        y = self.g_a(x)
        #print(y.shape)
        y_hat, y_likelihoods = self.factorized_entropy(y)
        x_hat = self.g_s(y_hat)


        return x_hat, y_likelihoods



if __name__ == '__main__':
    comp = Compressor(64, 128)
    loader = get_loader()
    x_hat, rx, rz = comp(x)

    print(x_hat.shape, rx, rz)
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

class FactorizedPrior(nn.Module):
    '''
    A Factorized entropy model for estimating the entropy of a representation
    '''
    def __init__(self, z_dim):
        super().__init__()
        self.factorized_entropy = EntropyBottleneck(z_dim)
        
    def forward(self, z):
        z_num = z.shape[0] * z.shape[1]
        z_hat, z_llh = self.factorized_entropy(z)
        rate_z =  (torch.sum(-1.0*torch.log2(z_llh)) / z_num)

        return rate_z


class HyperPrior(nn.Module):
  def __init__(self, N, M):
    super().__init__()
    self.hyper_encoder = nn.Sequential(
        nn.Linear(N, M),
        #nn.LeakyReLU(inplace=True),
        #nn.Linear(N,N),
        #nn.LeakyReLU(inplace=True),
        #nn.Linear(N,M),
    )
    self.mean = nn.Sequential(
        nn.Linear(M, N),
        nn.LeakyReLU(inplace=True),
        #nn.Linear(N,N),
        #nn.LeakyReLU(inplace=True),
        #nn.Linear(N,N),
        #nn.LeakyReLU(inplace=True),
    )
    self.std = nn.Sequential(
        nn.Linear(M, N),
        nn.LeakyReLU(inplace=True),
        #nn.Linear(N,N),
        #nn.LeakyReLU(inplace=True),
        #nn.Linear(N,N),
        #nn.LeakyReLU(inplace=True),
    )

    self.factorized = EntropyBottleneck(M)
    self.gaussian_conditional = GaussianConditional(None)
    self.N, self.M = N, M

  def forward(self, x):
    B= x.shape[0]
    x_num = B * self.N
    z_num = B * self.M

    z = self.hyper_encoder(x.flatten(1))
    z_hat, z_llh = self.factorized(z.unsqueeze(2).unsqueeze(2))
    mu = self.mean(z_hat.flatten(1)).unsqueeze(2).unsqueeze(2)
    sigma = self.std(z_hat.flatten(1)).unsqueeze(2).unsqueeze(2)
    x_hat, x_llh = self.gaussian_conditional(x, scales=sigma, means=mu)

    rate_x =  (torch.sum(-1.0*torch.log2(x_llh)) / x_num)
    rate_z =  (torch.sum(-1.0*torch.log2(z_llh)) / z_num)

    #print(rate_x, rate_z, self.training)
    #print(x[0, :5], x_hat[0, :5], z[0, :5], z_hat[0, :5], self.training)
    #print(rate_x.item(), rate_z.item())
    #print(self.training, z_hat[0, :5], mu[0, :5])
    return rate_x + rate_z
'''
class HyperPrior(nn.Module):
  def __init__(self, N, M):
    super().__init__()
    self.hyper_encoder = nn.Sequential(
        conv(N, M, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        conv(M, M, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        conv(M, M, stride=1, kernel_size=1),
    )
    self.gaussian_param = nn.Sequential(
        conv(M, N, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        conv(N, N * 3//2, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        conv(N * 3//2, N * 2, stride=1, kernel_size=1),
    )


    self.factorized = EntropyBottleneck(M)
    self.gaussian_conditional = GaussianConditional(None)
    self.N, self.M = N, M
  def forward(self, x):
    B= x.shape[0]
    x_num = B * self.N
    z_num = B * self.M

    z = self.hyper_encoder(x)
    z_hat, z_llh = self.factorized(z)
    gaus_param = self.gaussian_param(z_hat)

    sigma, mu = gaus_param.chunk(2, 1)

    x_hat, x_llh = self.gaussian_conditional(x, scales=sigma, means=mu)

    rate_x =  (torch.sum(-1.0*torch.log2(x_llh)) / x_num)
    rate_z =  (torch.sum(-1.0*torch.log2(z_llh)) / z_num)

    #print(rate_x, rate_z, self.training)
    #print(x[0, :5], x_hat[0, :5], z[0, :5], z_hat[0, :5], self.training)

    return rate_x + rate_z

'''


class SmallHyperPrior(nn.Module):
  def __init__(self, N, M):
    super().__init__()
    self.hyper_encoder = nn.Sequential(
        nn.Linear(N, M),
        nn.LeakyReLU(inplace=True),
        #nn.LeakyReLU(inplace=True),
        #nn.Linear(N,N),
        #nn.LeakyReLU(inplace=True),
        #nn.Linear(N,M),
    )
    self.hyper_decoder = nn.Sequential(
        nn.Linear(M, N*2),
        nn.LeakyReLU(inplace=True),
        #nn.Linear(N,N * 2),
        #nn.LeakyReLU(inplace=True),
        #nn.Linear(N,N),
        #nn.LeakyReLU(inplace=True),
    )

    self.factorized = EntropyBottleneck(M)
    self.gaussian_conditional = GaussianConditional(None)
    self.N, self.M = N, M

  def forward(self, x):
    B= x.shape[0]
    x_num = B * self.N
    z_num = B * self.M

    z = self.hyper_encoder(x.flatten(1))
    z_hat, z_llh = self.factorized(z.unsqueeze(2).unsqueeze(2))

    mu_sigma = self.hyper_decoder(z_hat.flatten(1)).unsqueeze(2).unsqueeze(2)
    mu, sigma = mu_sigma.chunk(2, 1)
    x_hat, x_llh = self.gaussian_conditional(x, scales=sigma, means=mu)

    rate_x =  (torch.sum(-1.0*torch.log2(x_llh)) / x_num)
    rate_z =  (torch.sum(-1.0*torch.log2(z_llh)) / z_num)

    #print(rate_x, rate_z, self.training)
    #print(x[0, :5], x_hat[0, :5], z[0, :5], z_hat[0, :5], self.training)
    #print(rate_x.item(), rate_z.item())
    #print(self.training, z_hat[0, :5], mu[0, :5])
    return rate_x + rate_z



if __name__ == '__main__':
    #comp = Compressor(64, 128)
    hyper = HyperPrior()
    loader = get_loader()
    x_hat, rx, rz = comp(x)

    print(x_hat.shape, rx, rz)
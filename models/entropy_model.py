import torch
from torch import nn
from .entropy.bottleneck import EntropyBottleneck, Gaussian_Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def conv(cin, cout, ksize, stride):
    return nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=ksize//2)

def deconv(cin, cout, ksize, stride):
    pad = ksize//2
    return nn.ConvTranspose2d(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=pad, output_padding=pad-1)


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
 
        z_hat, z_bits = self.factorized_entropy(z.unsqueeze(2).unsqueeze(3))
        rate_z =  z_bits / (z.shape[0], z.shape[1])

        return z_hat.squeeze(0).squeeze(0), rate_z



if __name__ == '__main__':
    comp = Compressor(128, 192)
    x = torch.ones([4, 128, 1, 1])
    x_hat, rx, rz = comp(x)

    print(x_hat.shape, rx, rz)
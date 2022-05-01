import torch
from torch import nn
from entropy.bottleneck import EntropyBottleneck, Gaussian_Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        
        noise = torch.rand_like(y) - 0.5
        y_hat = y + noise.to(device)

        z = self.hyper_encoder(y.squeeze(2).squeeze(2))
        z_hat, z_bits = self.factorized_entropy(z.unsqueeze(2).unsqueeze(3))

        hyper_out = self.hyper_decoder(z_hat.squeeze(2).squeeze(2))

        mu = self.mean_head(hyper_out).unsqueeze(2).unsqueeze(2)
        sigma = self.std_head(hyper_out).unsqueeze(2).unsqueeze(2)

        y_bits = self.gaussian_entropy(y, sigma, mu)

        rate_y, rate_z = y_bits/y.shape[1] , z_bits/z.shape[1]

        return y_hat, rate_y, rate_z



if __name__ == '__main__':
    comp = Compressor(128, 192)
    x = torch.ones([4, 128, 1, 1])
    x_hat, rx, rz = comp(x)

    print(x_hat.shape, rx, rz)
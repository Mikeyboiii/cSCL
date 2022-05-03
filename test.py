from models.entropy_model import AECompress
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from data.dataloader import get_loader
import torch
import math


loader, _ = get_loader('cifar10', 'train', normalize=True, views=2, bs=128, dl=False)
model = AECompress(64, 512).cuda()
    
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


for _, (data, _) in enumerate(loader):

    x =torch.cat(data, dim=0).cuda()

    model.train()
    x_hat, likelihood = model(x)


    rate = torch.sum(-1.0*torch.log2(likelihood)) / (x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])
    mse = criterion(x, x_hat)
    loss = 0 * mse + 128 * rate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(rate.item())
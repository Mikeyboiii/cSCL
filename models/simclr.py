import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    def __init__(self, z_dim=256, arch='resnet50'):
        super().__init__()
        assert arch in ['resnet50', 'resnet18']
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

    def forward(self, x):
        
        h = self.encoder(x)
        if self.training:
            h = h + torch.rand_like(h).to(device) - 0.5
        else:
            h = torch.round(h)

        z = self.projector(h)
        return h, z
        

if __name__ == '__main__':
    #x = torch.rand([4, 3, 224, 224])
    #y = torch.rand([4, 3, 224, 224])
    #model = simclr(arch='resnet18')
    #zx, zy = model(torch.cat([x, y], dim=0)).chunk(2, dim=0)
    #print(zx.shape, zy.shape)


    print(encoder(x).shape)
    #loss = nt_xent_loss(zx, zy, 0.1, eps=1e-6)

    #print(loss)


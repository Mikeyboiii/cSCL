import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


def nt_xent_loss(out_1, out_2, temperature, eps=1e-6):
    """
    assume out_1 and out_2 are normalized
    out_1: [batch_size, dim]
    out_2: [batch_size, dim]
    """
    # gather representations in case of distributed training
    # out_1_dist: [batch_size * world_size, dim]
    # out_2_dist: [batch_size * world_size, dim]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        out_1_dist = SyncFunction.apply(out_1)
        out_2_dist = SyncFunction.apply(out_2)
    else:
        out_1_dist = out_1
        out_2_dist = out_2

    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss


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
        if arch == 'resnet50':
            backbone = resnet50(pretrained=False)
            c_in = 2048
        elif arch == 'resnt18':
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

    #def forward(self, x):
    #    
    #    h = self.encoder(x)
    #    z = self.projector(h)
    #    return h.squeeze(2).squeeze(2), z
        

if __name__ == '__main__':
    x = torch.rand([4, 3, 224, 224])
    y = torch.rand([4, 3, 224, 224])
    model = simclr(arch='resnet50')

    hx = model.encoder(x)
    zx = model.projector(hx)

    hy = model.encoder(y)
    zy = model.projector(hy)

    loss = nt_xent_loss(zx, zy, 0.1, eps=1e-6)

    print(loss)


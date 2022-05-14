# Use an image compressor to pretrain entropy model
import torch
import torch.nn as nn
from models.nets import AE_fc
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import argparse

root = '~/cSCL'
train_set = MNIST(root = root + '/data/mnist', train = True, transform=ToTensor(), download=True)
loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)


def train(args):
    model = AE_fc().cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        train_loss, train_D, train_R = 0, 0, 0

        for iter, (x, _) in enumerate(loader):
            model.train()
            x = (2 * x - 1).cuda()

            x_hat, R = model(x)
            D = criterion(x_hat, x)

            loss = D + args.beta * R

            train_loss += loss.item()
            train_D += D.item()
            train_R += R.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('EP%d |train_loss=%.4f |train_D=%.4f| train_R=%.4f' %(ep, train_loss/len(loader), train_D/len(loader), train_R/len(loader))) 
        if (ep+1)%args.save_freq == 0:
            encoder_state = {'model': model.encoder.state_dict()}
            encoder_path = args.save_dir + 'Encoder_b%.3f_ep%d'%(args.beta, ep)+'.pkl'
            torch.save(encoder_state, encoder_path)

            entropy_state = {'model': model.entropy_model.state_dict()}
            entropy_path = args.save_dir + 'Factorized_b%.3f_ep%d'%(args.beta, ep)+'.pkl'
            torch.save(entropy_state, entropy_path)

if __name__ == '__main__':
    root = '/root/cSCL'

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default=root + '/pretrained_models/')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--beta', type=float, default=0.5, help='lagrangian multiplier')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--save_freq', type=int, default=20, help='frequency of saving model')


    args = parser.parse_args()

    train(args)

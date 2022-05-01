import torch
import argparse
from time import time
from data.dataloader import get_loader
from models.simclr import simclr
from losses import loss_fn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(args):
    assert args.dataset in ['cifar10']

    train_loader, _ = get_loader(args.dataset, 'train', normalize=True, views=2, bs=args.bs_train, dl=False)
    test_loader, _ = get_loader(args.dataset, 'test', normalize=True, views=2, bs=args.bs_test, dl=False)

    if args.dataset=='cifar10':
        C, H, W = 3, 32, 32
        num_classes = 10

    model = simclr(z_dim=args.z_dim, arch=args.arch)
    model = torch.nn.DataParallel(model)
    if args.pretrained is not None:
        model.load_state_dict(torch.load(args.pretrained)['model'])

    model.to(device)
    model.train()
    
    criterion = loss_fn(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    tic = time()

    for ep in range(args.epochs):
        train_loss, test_loss = 0, 0
        for iter, (imgs, labels) in enumerate(train_loader):
            model.train()

            z1, z2 = model(torch.cat(imgs, dim=0).to(device)).chunk(2, dim=0)

            loss = criterion(z1, z2, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.inference_mode():

            for iter, (imgs, labels) in enumerate(test_loader):
                model.eval()

                z1, z2 = model(torch.cat(imgs, dim=0).to(device)).chunk(2, dim=0)

                loss = criterion(z1, z2, labels)
                test_loss += loss.item()

        print('EP%d |train_loss=%.6f |test_loss=%.6f'%(ep, train_loss/args.bs_train, 5 * test_loss/args.bs_test))

        if (ep+1)%args.save_freq==0:
            state = {'model': model.module.state_dict()} if isinstance(model, torch.nn.DataParallel) else {'model': model.state_dict()} 
            path = args.save_dir + '%s_%s_t%.3f_ep%d'%(args.arch, args.dataset, args.temp, ep)+'.pkl'
            torch.save(state, path)
    toc = time()
    print('Time Elapsed: %dmin' %((toc-tic)//60))



if __name__ == '__main__':
    root = '/home/lz2814_columbia_edu/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--save_dir', type=str, default=root + '/pretrained_models/')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--arch', type=str, default='resnet18')

    parser.add_argument('--loss_type', type=str, default='cont', help='cont, supcont, c_cont, c_supcont')
    parser.add_argument('--temp', type=float, default=0.07, help='contrastive loss temperature')
    parser.add_argument('--beta', type=float, default=128, help='lagrangian multiplier')

    parser.add_argument('--z_dim', type=int, default=256)

    parser.add_argument('--bs_train', type=int, default=256, help='training batchsize')
    parser.add_argument('--bs_test', type=int, default=256, help='testing batchsize')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--save_freq', type=int, default=10, help='frequency of saving model')

    args = parser.parse_args()

    train(args)

import torch
import argparse
from time import time
from data.dataloader import get_loader
from models.nets import simclr
from losses import SupConLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(args):
    print('\nTraining: Dataset: %s |arch: %s |loss_type:%s |temp=%.3f |beta=%.3f| compress_rep:%s| z_dim=%d| lr=%.4f' %(
        args.dataset, args.arch, args.loss_type, args.temp, args.beta, args.compress_rep, args.z_dim, args.lr
    ))
    assert args.dataset in ['cifar10']

    if args.loss_type in ['c_cont', 'c_supcont']:
        ent_model = args.entropy_model
    else:
        ent_model= None

    train_loader, _ = get_loader(args.dataset, 'train', normalize=True, views=2, bs=args.bs_train, dl=True)
    test_loader, _ = get_loader(args.dataset, 'test', normalize=True, views=2, bs=args.bs_test, dl=True)

    model = simclr(z_dim=args.z_dim, arch=args.arch, entropy_model=ent_model)
    model = torch.nn.DataParallel(model)
    if args.pretrained is not None:
        model.module.load_state_dict(torch.load(args.pretrained)['model'])

    model.to(device)
    model.train()
    
    criterion = SupConLoss(temperature=args.temp)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [400])

    tic = time()

    for ep in range(args.epochs):
        train_loss, test_loss, train_cont, train_rate, test_cont, test_rate = 0, 0, 0, 0, 0, 0
        for iter, (imgs, labels) in enumerate(train_loader):
            model.train()


            z, rate = model(torch.cat(imgs, dim=0).to(device))
            z1, z2 = z.chunk(2, dim=0)
            batch = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)

            if 'sup' not in args.loss_type:
                cont = criterion(batch)
            else:
                cont = criterion(batch, labels)

            if ep < 200:
                loss = cont
            else:
                loss = cont + args.beta * rate
            #loss = cont + args.beta * rate
            train_loss += loss.item()
            train_cont += cont.item()
            train_rate += rate.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.inference_mode():

            for iter, (imgs, labels) in enumerate(test_loader):
                model.eval()

                z, rate = model(torch.cat(imgs, dim=0).to(device))
                z1, z2 = z.chunk(2, dim=0)
                batch = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)

                if 'sup' not in args.loss_type:
                    cont = criterion(batch)
                else:
                    cont = criterion(batch, labels)
                loss = cont + args.beta * rate
                test_loss += loss.item()
                test_cont += cont.item()
                test_rate += rate.item()

        print('EP%d |train_loss=%.6f |test_loss=%.6f| train_cont=%.6f| train_bits=%.6f| test_cont=%.6f| test_bits=%.6f'
            %(ep, train_loss/args.bs_train, 5 * test_loss/args.bs_test, 
            train_cont/args.bs_train, train_rate/args.bs_train,
            5 * test_cont/args.bs_test, 5 * test_rate/args.bs_test
            ))

        if (ep+1)%args.save_freq==0:
            state = {'model': model.module.state_dict()} if isinstance(model, torch.nn.DataParallel) else {'model': model.state_dict()} 
            path = args.save_dir + '%s_%s_%s_b%.3f_t%.3f_ep%d'%(args.arch, args.dataset, args.loss_type, args.beta, args.temp, ep)+'.pkl'
            torch.save(state, path)
            
        scheduler.step()
    toc = time()
    print('Time Elapsed: %dmin' %((toc-tic)//60))


if __name__ == '__main__':
    root = '/home/lz2814_columbia_edu/lingyu'

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--save_dir', type=str, default=root + '/pretrained_models/')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--entropy_model', type=str, default='hyperprior')

    parser.add_argument('--loss_type', type=str, default='cont', help='cont, supcont, c_cont, c_supcont, ce, c_ce')
    parser.add_argument('--temp', type=float, default=0.07, help='contrastive loss temperature')
    parser.add_argument('--beta', type=float, default=0.5, help='lagrangian multiplier')
    parser.add_argument('--compress_rep', type=str, default='h', help='compress h or z')

    parser.add_argument('--z_dim', type=int, default=128)

    parser.add_argument('--bs_train', type=int, default=256, help='training batchsize')
    parser.add_argument('--bs_test', type=int, default=256, help='testing batchsize')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--save_freq', type=int, default=100, help='frequency of saving model')


    args = parser.parse_args()


    args.pretrained = '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_supcont_b0.500_t0.070_ep299.pkl'

    train(args)

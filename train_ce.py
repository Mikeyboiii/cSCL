import torch
import argparse
from time import time
from data.dataloader import get_loader
from models.nets import c_resnet, c_resnet_mid

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(args):
    print('\nTraining: Dataset: %s |arch: %s |loss_type:%s |beta=%.3f| lr=%.4f' %(
        args.dataset, args.arch, args.loss_type, args.beta, args.lr
    ))
    assert args.dataset in ['cifar10']
    if args.dataset=='cifar10':
        num_classes = 10

    if args.loss_type not in ['c_ce']:
        ent_model= None
    else:
        ent_model = args.entropy_model

    train_loader, _ = get_loader(args.dataset, 'train', normalize=True, views=1, bs=args.bs_train, dl=True)
    test_loader, _ = get_loader(args.dataset, 'test', normalize=True, views=1, bs=args.bs_test, dl=True)

    #model = c_resnet(num_classes=num_classes, arch=args.arch, entropy_model=ent_model)
    model = c_resnet_mid(num_classes=num_classes, arch=args.arch, entropy_model=ent_model)
    model = torch.nn.DataParallel(model)
    if args.pretrained is not None:
        model.module.load_state_dict(torch.load(args.pretrained)['model'])

    model.to(device)
    model.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    tic = time()

    for ep in range(args.epochs):
        train_loss, test_loss, train_ce, train_rate, test_ce, test_rate = 0, 0, 0, 0, 0, 0
        train_correct, train_total = 0, 0
        for iter, (imgs, labels) in enumerate(train_loader):
            model.train()
            x = imgs.to(device)
            labels = labels.to(device)

            logits, rate = model(x)
            ce = criterion(logits, labels)
            preds = torch.argmax(logits, 1)

            if ep < 5:
                loss = ce
            else:
                loss = ce + args.beta * rate
            train_loss += loss.item()
            train_ce += ce.item()
            train_rate += rate.item()

            train_correct += (preds==labels).sum()
            train_total += x.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.inference_mode():
            correct, total = 0, 0
            for iter, (imgs, labels) in enumerate(test_loader):
                model.eval()

                x = imgs.to(device)
                labels = labels.to(device)

                logits, rate = model(x)
                ce = criterion(logits, labels)
                preds = torch.argmax(logits, 1)

                if ep < 5:
                    loss = ce
                else:
                    loss = ce + args.beta * rate

                test_loss += loss.item()
                test_ce += ce.item()
                test_rate += rate.item()

                correct += (preds==labels).sum()
                total += x.shape[0]

        scheduler.step()

        print('EP%d |train_loss=%.4f |test_loss=%.4f| train_ce=%.4f| train_bits=%.4f| test_ce=%.4f| test_bits=%.4f| train_acc=%.2f |test_acc=%.2f'
            %(ep, train_loss/args.bs_train, 5 * test_loss/args.bs_test, 
            train_ce/args.bs_train, train_rate/args.bs_train,
            5 * test_ce/args.bs_test, 5 * test_rate/args.bs_test, (100*train_correct)/train_total ,(100*correct)/total
            ))

        if (ep+1)%args.save_freq==0:
            state = {'model': model.module.state_dict()} if isinstance(model, torch.nn.DataParallel) else {'model': model.state_dict()} 
            path = args.save_dir + 'mid%s_%s_%s_b%.3f_ep%d'%(args.arch, args.dataset, args.loss_type, args.beta, ep)+'.pkl'
            torch.save(state, path)
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

    parser.add_argument('--loss_type', type=str, default='ce', help='ce, c_ce')
    parser.add_argument('--beta', type=float, default=0.1, help='lagrangian multiplier')

    parser.add_argument('--bs_train', type=int, default=128, help='training batchsize')
    parser.add_argument('--bs_test', type=int, default=128, help='testing batchsize')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--save_freq', type=int, default=100, help='frequency of saving model')

    args = parser.parse_args()
    train(args)
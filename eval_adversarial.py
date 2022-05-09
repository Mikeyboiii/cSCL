import torch
import argparse
from time import time
from data.dataloader import get_loader
from models.simclr import c_resnet
from attacks.pgd import PGD_attack

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval(args):
    assert args.dataset in ['cifar10']
    if args.dataset=='cifar10':
        num_classes = 10

    test_loader, info = get_loader(args.dataset, 'test', normalize=False, views=1, bs=args.bs_test, dl=True)
    def normalize(x):
        mu, sigma =torch.tensor(info['mean']).view(1, 3, 1, 1).to(device), torch.tensor(info['std']).view(1, 3, 1, 1).to(device)
        return (x - mu) / sigma

    if args.loss_type not in ['c_ce']:
        ent_model= None
    else:
        ent_model = args.entropy_model

    model = c_resnet(num_classes=num_classes, arch=args.arch, entropy_model=ent_model)
    #model = c_CNN()
    model = torch.nn.DataParallel(model)
    model.module.load_state_dict(torch.load(args.pretrained)['model'])
    model.to(device)
    model.eval()

    with torch.inference_mode():
        correct, total, test_rate = 0, 0, 0
        for iter, (imgs, labels) in enumerate(test_loader):

            print('Testing %.2f%%' %((100*iter)/len(test_loader)), end='\r')

            model.eval()

            #print(imgs.max(), imgs.min())
            x = imgs.to(device)
            labels = labels.to(device)

            logits, rate = model(normalize(x))
            preds = torch.argmax(logits, 1)

            test_rate += rate.item()

            correct += (preds==labels).sum()
            total += x.shape[0]

        print('bpp=%.6f| test_acc=%.2f'%(test_rate/args.bs_test, (100*correct)/total))

def eval_adv(args):
    assert args.dataset in ['cifar10']
    if args.dataset=='cifar10':
        num_classes = 10

    test_loader, info = get_loader(args.dataset, 'test', normalize=False, views=1, bs=args.bs_test, dl=True)
    def normalize(x):
        mu, sigma =torch.tensor(info['mean']).view(1, 3, 1, 1).to(device), torch.tensor(info['std']).view(1, 3, 1, 1).to(device)
        return (x - mu) / sigma

    if args.loss_type not in ['c_ce']:
        ent_model= None
    else:
        ent_model = args.entropy_model

    model = c_resnet(num_classes=num_classes, arch=args.arch, entropy_model=ent_model)
    model = torch.nn.DataParallel(model)
    model.module.load_state_dict(torch.load(args.pretrained)['model'])
    model.to(device)
    model.eval()

    Loss = torch.nn.CrossEntropyLoss()

    correct, total, test_rate = 0, 0, 0
    for iter, (imgs, labels) in enumerate(test_loader):

        print('Testing %.2f%%' %((100*iter)/len(test_loader)), end='\r')

        model.eval()

        x = imgs.to(device)
        labels = labels.to(device)

        adv = PGD_attack(x, labels, model, Loss, args.epsilon, args.steps, args.step_size, innormalize=normalize)

        with torch.inference_mode():
            logits, rate = model(normalize(x + adv))
        preds = torch.argmax(logits, 1)

        test_rate += rate.item()

        correct += (preds==labels).sum()
        total += x.shape[0]

    print('bpp=%.6f| test_acc=%.2f'%(test_rate/args.bs_test, (100*correct)/total))


if __name__ == '__main__':
    root = '/home/lz2814_columbia_edu/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--entropy_model', type=str, default='hyperprior')

    parser.add_argument('--loss_type', type=str, default='ce', help='ce, c_ce')
    parser.add_argument('--bs_test', type=int, default=5000, help='testing batchsize')

    parser.add_argument('--epsilon', type=float, default=8/255, help='pgd bound')
    parser.add_argument('--steps', type=int, default=7, help='pgd steps')
    parser.add_argument('--step_size', type=float, default=2/255, help='pgd step size')


    args = parser.parse_args()

    args.pretrained = '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_ce_b0.100_ep329.pkl'


    #eval(args)
    eval_adv(args)
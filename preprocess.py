from torchvision import transforms


def simclr_aug(args):
    if args.dataset == 'cifar10':
        img_size = 32

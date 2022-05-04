#adapted from https://github.com/ae-foster/pytorch-simclr

import torch
import torchvision
from torchvision import transforms
from PIL import Image

root = '/home/lz2814_columbia_edu/'

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])

    return color_distort

def get_gaussian_blur(im_size, p=0.5):
    kernel_size = im_size//10
    transform = transforms.RandomApply(torch.nn.ModuleList([
        transforms.GaussianBlur(kernel_size),
        ]), p=p)
    return transform


def get_preprocess(dataset, split, views):
    if dataset == 'cifar10':
        img_size = 32

    if views==1: return [transforms.ToTensor()]

    if split == 'train':
        preprocess = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            get_color_distortion(),
            get_gaussian_blur(img_size),
            transforms.ToTensor(),
        ]

    elif split == 'test':
        preprocess = [
            transforms.ToTensor(),
        ]

    return preprocess

class cifar10_pair_data(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        pil_img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(pil_img)
            img2 = self.transform(pil_img)
        else:
            img2 = img = pil_img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, img2), target#, index 


def get_loader(dataset, split, normalize, bs, views=2, dl=False, augment=True):
    assert views==2 or views==1
    if dataset == 'cifar10':
        info = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2471, 0.2435, 0.2616)}
        set_dir = root + '/data/cifar10/'
        if views==2:
            set_class = cifar10_pair_data
        else:
            set_class = torchvision.datasets.CIFAR10
        
    trans = []
    
    if augment:
        trans += get_preprocess(dataset, split, views)

    if normalize: 
        trans.append(transforms.Normalize(info['mean'], info['std']))

    if split == 'train':
        loader = torch.utils.data.DataLoader(
        set_class(set_dir, train=True, download=dl,
                                    transform=torchvision.transforms.Compose(trans)),
                                    batch_size=bs, shuffle=True)
    elif split == 'test':
        loader = torch.utils.data.DataLoader(
        set_class(set_dir, train=False, download=dl,
                                    transform=torchvision.transforms.Compose(trans)),
                                    batch_size=bs, shuffle=False)

    return loader, info 
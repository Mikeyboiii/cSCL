import torch
import torchvision
from PIL import Image

root = ''
download = False

class cifar10_data_self(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(pil_img)
            img2 = self.transform(pil_img)
        else:
            img2 = img = pil_img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, img2), target#, index 



def get_loader(dataset, split, nonormalize, bs, dl=False):
    if dataset == 'cifar10':
        trans = [torchvision.transforms.ToTensor()]
        if nonormalize == False: 
            trans.append(torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)))

        if split == 'train':
            loader = torch.utils.data.DataLoader(
            cifar10_data_self(root + '/data/', train=True, download=dl,
                                        transform=torchvision.transforms.Compose(trans)),
                                        batch_size=bs, shuffle=True)
        elif split == 'test':
            loader = torch.utils.data.DataLoader(
            cifar10_data_self(root + '/data/', train=False, download=dl,
                                        transform=torchvision.transforms.Compose(trans)),
                                        batch_size=bs, shuffle=False)

        info = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2471, 0.2435, 0.2616]}

    return loader, info 
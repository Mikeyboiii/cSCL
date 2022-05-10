from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from models.nets import c_resnet_mid, c_resnet
from data.dataloader import get_loader
import torch
import matplotlib.pyplot as plt



test_loader, info = get_loader('cifar10', 'test', normalize=False, views=1, bs=1, dl=False)
def normalize(x):
    mu, sigma =torch.tensor(info['mean']).view(1, 3, 1, 1).cuda(), torch.tensor(info['std']).view(1, 3, 1, 1).cuda()
    return (x - mu) / sigma


model = c_resnet_mid(num_classes=10, arch='resnet18')
model = torch.nn.DataParallel(model)
model.module.load_state_dict(torch.load('/home/lz2814_columbia_edu/lingyu/pretrained_models/midresnet18_cifar10_c_ce_b0.200_ep99.pkl')['model'])

#model = c_resnet(num_classes=10, arch='resnet18', entropy_model=None)
#model = torch.nn.DataParallel(model)
#model.module.load_state_dict(torch.load('/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_ce_b0.100_ep99.pkl')['model'])


model.cuda()
model.eval()


img, _ = next(iter(test_loader))


#f = model.module.encoder[0](normalize(img.cuda()))
#f = model.module.encoder[1](f)
#f = model.module.encoder[2](f)
#f = model.module.encoder[3](f)
#f = model.module.encoder[4](f)
_, _, f = model(normalize(img.cuda()))

images = []
for i in range(64):
    f_i = f[0, i, :, :].detach().cpu().numpy()
    images.append(f_i)
    #plt.imsave('fmaps/fmap_%d.png'%i, f_i)
#print(f.shape)


fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(8,8))
for idx, image in enumerate(images):
    row = idx // 8
    col = idx % 8
    axes[row, col].axis("off")
    axes[row, col].imshow(image,  aspect="auto")
plt.subplots_adjust(wspace=.05, hspace=.05)
plt.savefig('comp.png')
import torch
from tools.gradcam.gradcam import GradCAMpp
from tools.gradcam.utils import visualize_cam
from data.dataloader import get_loader
from models.nets import c_resnet
from torchvision.models import resnet18
import matplotlib.pyplot as plt

#mode = 'cce'
mode = 'ce'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_loader, info = get_loader('cifar10', 'test', normalize=False, views=1, bs=1, dl=False)
'''
betas = [0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32]

paths = [
    '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_c_ce_b0.050_ep99.pkl',
    '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_c_ce_b0.100_ep99.pkl',
    '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_c_ce_b0.200_ep99.pkl',
    '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_c_ce_b0.500_ep99.pkl',
    '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_c_ce_b1.000_ep99.pkl',
    '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_c_ce_b2.000_ep99.pkl',
    '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_c_ce_b4.000_ep99.pkl',
    '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_c_ce_b8.000_ep99.pkl',
    '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_c_ce_b16.000_ep99.pkl',
    '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_c_ce_b32.000_ep99.pkl',
]
'''
def normalize(x):
    mu, sigma =torch.tensor(info['mean']).view(1, 3, 1, 1).to(device), torch.tensor(info['std']).view(1, 3, 1, 1).to(device)
    return (x - mu) / sigma

if mode=='ce':
    model_path = '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_ce_b0.100_ep99.pkl'
    model = c_resnet(num_classes=10, arch='resnet18', entropy_model=None)
    save_path = 'ce_imgs'

elif mode=='cce':
    model_path = '/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_c_ce_b2.000_ep99.pkl'
    model = c_resnet(num_classes=10, arch='resnet18', entropy_model='hyperprior')
    save_path = 'com_imgs'

model.load_state_dict(torch.load(model_path)['model'])

model.to(device)
model.eval()


#print(model._modules['encoder'][7][-1])
model_dict = dict(type='resnet', arch=model, layer_name='last', input_size=(32, 32))

gradcampp = GradCAMpp(model_dict)



for i, (img, label) in enumerate(test_loader):
    img = img.to(device)


    mask, logit = gradcampp(normalize(img), class_idx=label)
    if torch.argmax(logit).item() != label:
        print(i, torch.argmax(logit).item(), label.item())

    heatmap, cam_result = visualize_cam(mask.cpu(), img)
    plt.imsave(save_path + '/grad_%d.png'%i, cam_result.permute(1,2,0).cpu().numpy())

    #plt.imsave( 'imgs/grad_%d.png'%i, img.squeeze(0).permute(1,2,0).cpu().numpy())
    #print(heatmap)

    if i == 100: break

'''

for i, p in enumerate(paths):
    print(i)
    model = c_resnet(num_classes=10, arch='resnet18', entropy_model='hyperprior')
    model.load_state_dict(torch.load(p)['model'])
    w = model.fc.weight
    s = torch.linalg.svdvals(w).cpu().detach().numpy()
    plt.plot(s, label='beta=%.2f'%betas[i], marker='.')

model = c_resnet(num_classes=10, arch='resnet18', entropy_model=None)
model.load_state_dict(torch.load('/home/lz2814_columbia_edu/lingyu/pretrained_models/resnet18_cifar10_ce_b0.100_ep99.pkl')['model'])
w = model.fc.weight
s = torch.linalg.svdvals(w).cpu().detach().numpy()
plt.plot(s, label='ce', marker='.')

plt.xlabel('ranking')
plt.ylabel('singular value')
plt.grid()
plt.legend()
plt.savefig('singular.png')

'''
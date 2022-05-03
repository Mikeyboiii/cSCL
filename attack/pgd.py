from torch.autograd import Variable
import torch

def norm2(v):
    v = v / (torch.sum(v**2, dim=1, keepdim=True)**0.5 + 1e-10)
    return v

upper_bound, lower_bound = 1,0
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def PGD_attack(x, y, net, Loss, epsilon, steps, step_size, innormalize=lambda x: x, norm = "l_inf"):
    '''
    Generates attacked image for a single task.
    :param x: 
    :param y: 
    :param net: 
    :param Loss: 
    :param epsilon: 
    :param steps: 
    :param dataset: 
    :param step_size: 
    :param info: 
    :param using_noise: 
    :return: 
    '''
    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True
    x_adv = x.clone()

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        x = x.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    delta = torch.zeros_like(x_adv).cuda()
    
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
        
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon
    else:
        print('error')
        exit(0)
    delta = clamp(delta, lower_bound-x_adv, upper_bound-x_adv)
    delta = Variable(delta, requires_grad=True)
    delta.requires_grad = True

    for i in range(steps):
        delta = Variable(delta.data, requires_grad=True)
        preds = net(innormalize(x_adv+delta))
        cost = Loss(preds, y) 
        net.zero_grad()
        cost.backward()

        if norm == "l_inf":
            delta.grad.sign_()
            delta = delta + step_size * delta.grad
            delta = torch.clamp(delta,  min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g = delta.grad.detach()
            d = delta
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d + scaled_g*step_size).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            delta = d
        
        delta = clamp(delta, lower_bound - x, upper_bound - x)

    return delta.data
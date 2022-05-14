import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from models.nets import MLP

root = '~/cSCL'

train_val = MNIST(root = root + '/data/mnist', train = True, transform=ToTensor(), download=True)
train_set, val_set = torch.utils.data.random_split(train_val, [50000, 10000])
test_set = MNIST(root = root + '/data/mnist', train = False,  transform=ToTensor(), download=True)

bs = 100
train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False)
final_loader = torch.utils.data.DataLoader(train_val, batch_size=bs, shuffle=True)

class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


def train(beta, final=False):
    if beta==0:
        model = MLP(ent_model=None).cuda()
    else: 
        model = MLP(ent_model='factorized').cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.97)

    ema = EMA(0.999)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    if final: 
        loader = final_loader
        loader2 = test_loader
    else:
        loader = train_loader
        loader2 = val_loader

    for ep in range(200):
        train_loss, val_loss, train_ce, train_rate, val_ce, val_rate = 0, 0, 0, 0, 0, 0
        train_correct, train_total = 0, 0

        for iter, (x, y) in enumerate(loader):
            model.train()
            x = (2 * x - 1).cuda()
            y = y.cuda()

            logits, rate = model(x)
            ce = criterion(logits, y)
            preds = torch.argmax(logits, 1)

            if ep < 0 or beta==0:
                loss = ce
            else:
                loss = ce + beta * rate

            train_loss += loss.item()
            train_ce += ce.item()
            train_rate += rate.item()

            train_correct += (preds==y).sum()
            train_total += x.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema(name, param.data)

        with torch.no_grad():
            correct, total = 0, 0
            
            if beta==0:
                model_avg =  MLP(ent_model=None).cuda()
            else:
                model_avg =  MLP(ent_model='factorized').cuda()
            for name, param in model_avg.named_parameters():
                if param.requires_grad:
                    param.data = ema.shadow[name]

            for iter, (x, y) in enumerate(loader2):
                model.eval()
                x = (2 * x - 1).cuda()
                y = y.cuda()

                logits, rate = model_avg(x)
                ce = criterion(logits, y)
                preds = torch.argmax(logits, 1)

                if ep < 0 or beta==0:
                    loss = ce
                else:
                    loss = ce + beta * rate

                val_loss += loss.item()
                val_ce += ce.item()
                val_rate += rate.item()

                correct += (preds==y).sum()
                total += x.shape[0]

        scheduler.step()

        if (ep+1)%1==0:
            print('EP%d |train_loss=%.2f |val_loss=%.2f| train_ce=%.2f| train_bits=%.4f| val_ce=%.2f| val_bits=%.4f| train_acc=%.2f |val_acc=%.2f| Error=%.2f' %(ep, train_loss, 5 * val_loss, train_ce, train_rate/len(loader),
            5 * val_ce,  val_rate /len(loader2), (100*train_correct)/train_total ,(100*correct)/total, 100-(100*correct)/total)) 
            


final = False
#for beta in [0]:
#for beta in [1, 1e-1, 1e-2]:
#for beta in [1e-3, 1e-4, 1e-5]:
for beta in [1e-2]:

    print('Beta =', beta, '| final =', final)
    train(beta=beta, final=final)



#
#base: 1.36

#-------------------
#base: 1.46
# 1e-8: 1.85, 1.60(100epoch), 
# 1e-7 1.88
# 1e-6 1.78
# 1e-5 1.50
# 1e-4 1.49
# 1e-3 1.39
# 1e-2 1.55
# 1e-1 1.79



# ------------------
# baseline: 1.56, 1.54

# 1e-7: 1.74
# 1e-6: 1.59
# 1e-5: 1.42
# 1e-4: 1.56, 1.52
# 1e-3: 1.50
# 1e-2: 1.55
# 1e-1: 1.82
# 2e-1: 2.26
# 5e-1: 1.84
# ------------------
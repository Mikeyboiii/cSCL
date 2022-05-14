import torch
from torch.nn import nn
from models.nets import AE_fc
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

root = '~/cSCL'
train_set = MNIST(root = root + '/data/mnist', train = True, transform=ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)


def train(args):
    model = AE_fc().cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        train_loss, val_loss, train_D, train_R, val_D, val_R = 0, 0, 0, 0, 0, 0

        for iter, (x, _) in enumerate(loader):
            model.train()
            x = (2 * x - 1).cuda()

            x_hat, rate = model(x)
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

if __name__ == '__main__':
    root = '~'

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default=root + '/pretrained_models/')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--beta', type=float, default=0.5, help='lagrangian multiplier')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_freq', type=int, default=100, help='frequency of saving model')


    args = parser.parse_args()

    train(args)

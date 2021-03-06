import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

from data.dataloader import get_loader
from models.nets import simclr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)

def generate_embeddings(loader, simclr_model):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.inference_mode():
            h = simclr_model.encoder(x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader):
    train_X, train_y = generate_embeddings(train_loader, simclr_model)
    test_X, test_y = generate_embeddings(test_loader, simclr_model)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(device).squeeze(2).squeeze(2)
        y = y.to(device)


        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()


    return loss_epoch, accuracy_epoch


def test(loader, model, criterion):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(device).squeeze(2).squeeze(2)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def eval(args):

    train_loader, _ = get_loader(args.dataset, 'train', normalize=True, views=1, bs=args.bs_emb, dl=False)
    test_loader, _ = get_loader(args.dataset, 'test', normalize=True, views=1, bs=args.bs_emb, dl=False)

    n_features = 2048 if args.arch=='resnet50' else 512


    if args.loss_type in ['c_cont', 'c_supcont']:
        ent_model = 'hyperprior'
    else:
        ent_model = None

    simclr_model = simclr(z_dim=args.z_dim, arch=args.arch, entropy_model=ent_model)
    simclr_model.load_state_dict(torch.load(args.model_path)['model'])
    simclr_model.to(device)

    simclr_model.eval()

    ## Logistic Regression
    if args.dataset in ['cifar10', 'stl10']:
        n_classes = 10 
    model = LogisticRegression(n_features, n_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(simclr_model, train_loader, test_loader)
    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(train_X, train_y, test_X, test_y, args.bs)


    for epoch in range(args.epochs):
        loss_epoch, accuracy_epoch = train_one_epoch(arr_train_loader, model, criterion, optimizer)
        if epoch%50==0:
            print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}")

    # final testing
    loss_epoch, accuracy_epoch = test(arr_test_loader, model, criterion)
    print(f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}")

    
if __name__ == "__main__":

    root = '/home/lz2814_columbia_edu/lingyu/'
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--z_dim', type=int, default=128)

    parser.add_argument('--loss_type', type=str, default='cont', help='cont, supcont, c_cont, c_supcont')
    parser.add_argument('--model_path', type=str, default=None, help='pretrained model path')
    parser.add_argument('--lr', type=float, default=3e-4)

    parser.add_argument('--bs_emb', type=int, default=4096, help='generate embedding')
    parser.add_argument('--bs', type=int, default=256, help='train & test linear batchsize')

    parser.add_argument('--epochs', type=int, default=500, help='linear model epochs')
    args = parser.parse_args()

    #args.model_path = root + 'pretrained_models/resnet18_cifar10_c_cont_b0.500_t0.070_ep299.pkl'
    args.model_path = root + 'pretrained_models/resnet18_cifar10_c_cont_b0.100_t0.070_ep399.pkl'
    eval(args)


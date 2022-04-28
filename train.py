from models.simclr import simclr, nt_xent_loss
from argparse import ArgumentParser


def train(args):
    pass







if __name__ == '__main__':
    parser = ArgumentParser()

    # model params
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    # specify flags to store false

    parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
    parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--online_ft", action="store_true")
    parser.add_argument("--fp32", action="store_true")

    # transform params
    parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
    parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
    parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
    parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

    # training params
    parser.add_argument("--fast_dev_run", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")


    parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

    parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
    parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

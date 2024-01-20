import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['cifar10', 'cifar100', 'tinyimagenet', 'boolq', 'qnli', 'rte', 'imagenet', 'food101'])
    parser.add_argument('--model_name', type=str, default='resnet18',
                        choices=['resnet18', 'wresnet', 'cvt', 'cvt_pretrained', 'bert'])

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args

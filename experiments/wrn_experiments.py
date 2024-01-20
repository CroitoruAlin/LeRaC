import torch
import torchvision

from data_handlers import get_train_loader_tiny_imagenet, get_valid_loader_tiny_imagenet, get_train_val_loaders_cifar, \
    get_test_loader_cifar, get_train_loader_imagenet_subset, get_val_loader_imagenet, get_train_val_loaders_food101, \
    get_test_loader_food101
from experiments.r18_experiments import build_optimizer_resnet, build_optimizer_resnet_baseline
from models.resnet_official import wide_resnet50_2
from models.wide_resnet import Wide_ResNet
from train.resnet_train import Trainer, TrainerBaseline


def add_optimizer_params_lr(trainer, args):
    possible_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    groups = {"layer1":[str(i) for i in range(3)],
              "layer2":[str(i) for i in range(4)],
              "layer3":[str(i) for i in range(6)],
              "layer4":[str(i) for i in range(3)],
              }

    dictionary_lr = {}
    start_lr = args.initial_learning_rate
    for layer in possible_layers:

        for i, group in enumerate(groups[layer]):
            dictionary_lr[layer+"."+group] = start_lr
            if start_lr > args.min_lr:
                start_lr *= 1e-1

    for name, param in trainer.model.named_parameters():
        if name.startswith('conv1') or name.startswith('bn1'):
            print(f"Setting lr for {name} to {args.initial_learning_rate}")
            trainer.optimizer.add_param_group({'params': param, 'lr': args.initial_learning_rate})

        elif name[:8] in dictionary_lr:
            print(f"Setting lr for {name} to {dictionary_lr[name[:8]]}")
            trainer.optimizer.add_param_group({'params': param, 'lr': dictionary_lr[name[:8]]})
        elif 'linear' not in name and not name.startswith("fc."):
            print(f"Setting lr for {name} to {args.initial_learning_rate *args.clf_lr }")
            trainer.optimizer.add_param_group({'params': param, 'lr': args.initial_learning_rate * args.clf_lr})


def train_cifar10(args):
    args.num_classes = 10
    args.update_epoch = 1
    args.decay_epoch = 38
    args.decay_step = 30
    args.stop_decay_epoch = 99
    args.stop_update_epoch = 20
    args.scale_lr = 10
    args.initial_learning_rate = 1e-1
    args.min_lr = 1e-7
    args.clf_lr = 1e-5
    args.num_epochs = 200
    args.update_per_epoch = 1
    args.stop_lr = 0.11

    wresnet = Wide_ResNet(52, 2, 0.3, num_classes=args.num_classes)
    train_loader, _ = get_train_val_loaders_cifar(val_size=0)
    val_loader = get_test_loader_cifar()
    trainer = Trainer(wresnet, train_loader, val_loader, add_optimizer_params_lr, args, build_optimizer_resnet)
    trainer.train()


def train_cifar100(args):
    args.num_classes = 100
    args.update_epoch = 1
    args.decay_epoch = 38
    args.decay_step = 30
    args.stop_decay_epoch = 99
    args.stop_update_epoch = 20
    args.scale_lr = 10
    args.initial_learning_rate = 1e-1
    args.min_lr = 1e-7
    args.clf_lr = 1e-5
    args.num_epochs = 200
    args.update_per_epoch = 1
    args.stop_lr = 0.11

    wresnet = Wide_ResNet(52, 2, 0.3, num_classes=args.num_classes)
    train_loader, _ = get_train_val_loaders_cifar(val_size=0, dataset=torchvision.datasets.CIFAR100)
    val_loader = get_test_loader_cifar(dataset=torchvision.datasets.CIFAR100)
    trainer = Trainer(wresnet, train_loader, val_loader, add_optimizer_params_lr, args,build_optimizer_resnet)
    trainer.train()


def train_tiny_imagenet(args):
    args.num_classes = 200
    args.update_epoch = 1
    args.decay_epoch = 32
    args.decay_step = 30
    args.stop_decay_epoch = 93
    args.stop_update_epoch = 10
    args.scale_lr = 10
    args.initial_learning_rate = 1e-1
    args.min_lr = 1e-5
    args.clf_lr = 1e-5
    args.num_epochs = 100
    args.update_per_epoch = 4
    args.stop_lr = 0.11
    wresnet = Wide_ResNet(52, 2, 0.3, num_classes=args.num_classes)
    train_loader, class_to_idx = get_train_loader_tiny_imagenet(path="./data")
    val_loader = get_valid_loader_tiny_imagenet(path="./data", class_to_idx=class_to_idx)
    trainer = Trainer(wresnet, train_loader, val_loader, add_optimizer_params_lr, args, build_optimizer=build_optimizer_resnet)
    trainer.train()


def train_imagenet(args):
    args.num_classes = 200
    args.update_epoch = 1
    args.decay_epoch = 32
    args.decay_step = 30
    args.stop_decay_epoch = 93
    args.stop_update_epoch = 10
    args.scale_lr = 10
    args.initial_learning_rate = 1e-1
    args.min_lr = 1e-5
    args.clf_lr = 1e-5
    args.num_epochs = 120
    args.update_per_epoch = 4
    args.stop_lr = 0.11
    wresnet = wide_resnet50_2()
    for name, _ in wresnet.named_parameters():
        print(name)
    train_loader, class_to_idx = get_train_loader_imagenet_subset()
    val_loader = get_val_loader_imagenet(class_to_idx=class_to_idx, subset=True)
    trainer = Trainer(wresnet, train_loader, val_loader, add_optimizer_params_lr, args, build_optimizer=build_optimizer_resnet)
    trainer.train()

def train_food101(args):
    args.num_classes = 101
    args.update_epoch = 1
    args.decay_epoch = 32
    args.decay_step = 30
    args.stop_decay_epoch = 93
    args.stop_update_epoch = 10
    args.scale_lr = 10
    args.initial_learning_rate = 1e-1
    args.min_lr = 1e-5
    args.clf_lr = 1e-5
    args.num_epochs = 150
    args.update_per_epoch = 4
    args.stop_lr = 0.11
    wresnet = Wide_ResNet(52, 2, 0.3, num_classes=args.num_classes)
    train_loader = get_train_val_loaders_food101()
    val_loader = get_test_loader_food101()
    trainer = Trainer(wresnet, train_loader, val_loader, add_optimizer_params_lr, args, build_optimizer=build_optimizer_resnet)
    trainer.train()

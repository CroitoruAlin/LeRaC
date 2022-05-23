import torchvision

from data_handlers import get_train_loader_tiny_imagenet, get_valid_loader_tiny_imagenet, get_train_val_loaders_cifar, \
    get_test_loader_cifar
from experiments.r18_experiments import build_optimizer_resnet
from models.wide_resnet import Wide_ResNet
from train.resnet_train import Trainer


def add_optimizer_params_lr(trainer, args):
    possible_layers = ['layer1', 'layer2', 'layer3']
    groups = [str(i) for i in range(8)]
    dictionary_lr = {}
    start_lr = args.initial_learning_rate
    for layer in possible_layers:

        for i, group in enumerate(groups):
            dictionary_lr[layer+"."+group] = start_lr
            if start_lr > args.min_lr:
                start_lr *= 1e-1

    for name, param in trainer.model.named_parameters():
        if 'conv1_initial' in name:
            print(f"Setting lr for {name} to {args.initial_learning_rate}")
            trainer.optimizer.add_param_group({'params': param, 'lr': args.initial_learning_rate})

        elif name[:8] in dictionary_lr:
            print(f"Setting lr for {name} to {dictionary_lr[name[:8]]}")
            trainer.optimizer.add_param_group({'params': param, 'lr': dictionary_lr[name[:8]]})
        elif 'linear' not in name:
            print(f"Setting lr for {name} to {args.initial_learning_rate *args.clf_lr }")
            trainer.optimizer.add_param_group({'params': param, 'lr': args.initial_learning_rate* args.clf_lr})


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

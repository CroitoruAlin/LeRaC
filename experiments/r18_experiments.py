import torchvision
from torch import optim

from data_handlers import get_train_val_loaders_cifar, get_test_loader_cifar, get_valid_loader_tiny_imagenet, \
    get_train_loader_tiny_imagenet, get_train_loader_imagenet, get_val_loader_imagenet, \
    get_train_loader_imagenet_subset, get_train_val_loaders_food101, get_test_loader_food101
from models.resnet import ResNet18
from train.resnet_train import Trainer, TrainerBaseline
from models.resnet_official import resnet18 as resnet_builder

def add_optimizer_params_lr(trainer, args):
    possible_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    groups = [str(i) for i in range(2)]
    dictionary_lr = {}
    start_lr = args.initial_learning_rate * 1e-1

    for layer in possible_layers:

        for i, group in enumerate(groups):
            dictionary_lr[layer + "." + group] = start_lr
            if start_lr > args.min_lr:
                start_lr *= 1e-1
    for name, param in trainer.model.named_parameters():
        if name.startswith('conv1'):
            print(f"Setting lr for {name} to {args.initial_learning_rate}")
            trainer.optimizer.add_param_group({'params': param, 'lr': args.initial_learning_rate})

        elif name[:8] in dictionary_lr:
            print(f"Setting lr for {name} to {dictionary_lr[name[:8]]}")
            trainer.optimizer.add_param_group({'params': param, 'lr': dictionary_lr[name[:8]]})
        elif 'linear' not in name and not name.startswith("fc."):
            print(f"Setting lr for {name} to {args.initial_learning_rate}")
            trainer.optimizer.add_param_group({'params': param, 'lr': args.initial_learning_rate})

def build_optimizer_resnet(model, args):
    list_params = []
    for name, param in model.named_parameters():
        if 'linear' in name or name.startswith("fc."):
            list_params.append(param)
    return optim.SGD(list_params,
                               lr=args.initial_learning_rate * args.clf_lr,
                               weight_decay=1e-4,
                               momentum=0.9,
                               )
def build_optimizer_resnet_baseline(model,args):

    return optim.SGD(model.parameters(),
                     lr=args.initial_learning_rate ,
                     weight_decay=1e-4,
                     momentum=0.9,
                     )
def train_cifar10(args):
    args.num_classes = 10
    args.update_epoch = 1
    args.decay_epoch = 35
    args.decay_step = 30
    args.stop_decay_epoch = 96
    args.stop_update_epoch = 20
    args.scale_lr = 10
    args.initial_learning_rate=1e-1
    args.min_lr=1e-5
    args.clf_lr=1e-3
    args.num_epochs=200
    args.update_per_epoch=1
    args.stop_lr=0.11
    resnet18 = ResNet18(num_classes=args.num_classes)
    train_loader, _ = get_train_val_loaders_cifar(val_size=0, dataset=torchvision.datasets.CIFAR10)
    val_loader = get_test_loader_cifar(dataset=torchvision.datasets.CIFAR10)
    trainer = Trainer(resnet18, train_loader, val_loader, add_optimizer_params_lr, args,build_optimizer_resnet)
    trainer.train()


def train_cifar100(args):
    args.num_classes = 100
    args.update_epoch = 1
    args.decay_epoch = 36
    args.decay_step = 30
    args.stop_decay_epoch = 97
    args.stop_update_epoch = 20
    args.scale_lr = 10
    args.initial_learning_rate = 1e-1
    args.min_lr = 1e-6
    args.clf_lr = 1e-5
    args.num_epochs = 200
    args.update_per_epoch = 1
    args.stop_lr = 0.11
    resnet18 = ResNet18(num_classes=args.num_classes)
    train_loader, _ = get_train_val_loaders_cifar(val_size=0, dataset=torchvision.datasets.CIFAR100)
    val_loader = get_test_loader_cifar(dataset=torchvision.datasets.CIFAR100)

    trainer = Trainer(resnet18, train_loader, val_loader, add_optimizer_params_lr, args,build_optimizer_resnet)
    trainer.train()



def train_tinyimagenet(args):
    args.num_classes = 200
    args.update_epoch = 1
    args.decay_epoch = 33
    args.decay_step = 30
    args.stop_decay_epoch = 94
    args.stop_update_epoch = 20
    args.scale_lr = 10
    args.initial_learning_rate = 1e-1
    args.min_lr = 1e-7
    args.clf_lr = 1e-7
    args.num_epochs = 100
    args.update_per_epoch = 4
    args.stop_lr = 0.11
    resnet18 = ResNet18(num_classes=args.num_classes, kernel_size_avg_pool=8)
    train_loader, class_to_idx = get_train_loader_tiny_imagenet(path="./data")
    val_loader = get_valid_loader_tiny_imagenet(path="./data", class_to_idx=class_to_idx)
    trainer = Trainer(resnet18, train_loader, val_loader, add_optimizer_params_lr, args, build_optimizer_resnet)
    trainer.train()

def train_imagenet(args):
    args.num_classes = 125
    args.update_epoch = 1
    args.decay_epoch = 33
    args.decay_step = 30
    args.stop_decay_epoch = 94
    args.stop_update_epoch = 20
    args.scale_lr = 10
    args.initial_learning_rate = 1e-1
    args.min_lr = 1e-7
    args.clf_lr = 1e-7
    args.num_epochs = 100
    args.update_per_epoch = 4
    args.stop_lr = 0.11
    resnet18 = resnet_builder()
    train_loader, class_to_idx = get_train_loader_imagenet_subset()
    val_loader = get_val_loader_imagenet(class_to_idx=class_to_idx, subset=True)
    trainer = Trainer(resnet18, train_loader, val_loader, add_optimizer_params_lr, args, build_optimizer_resnet)
    trainer.train()
def train_food101(args):
    args.num_classes = 101
    args.update_epoch = 1
    args.decay_epoch = 36
    args.decay_step = 30
    args.stop_decay_epoch = 97
    args.stop_update_epoch = 20
    args.scale_lr = 10
    args.initial_learning_rate = 1e-1
    args.min_lr = 1e-6
    args.clf_lr = 1e-5
    args.num_epochs = 200
    args.update_per_epoch = 1
    args.stop_lr = 0.11
    resnet18 = ResNet18(num_classes=args.num_classes)

    train_loader = get_train_val_loaders_food101()
    val_loader = get_test_loader_food101()
    trainer = Trainer(resnet18, train_loader, val_loader, add_optimizer_params_lr, args,
                      build_optimizer=build_optimizer_resnet)
    trainer.train()


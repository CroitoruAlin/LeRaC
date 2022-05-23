import torchvision
import yaml
from torch import optim

from data_handlers import get_train_val_loaders_cifar, get_test_loader_cifar, get_train_loader_tiny_imagenet, \
    get_valid_loader_tiny_imagenet
from models.cvt import ConvolutionalVisionTransformer
from train.cvt_train import TrainerCvt


def build_cvt(args):
    with open("configs/cvt_configs.yaml", 'r') as stream:
        try:
            data = (yaml.safe_load(stream))
        except Exception as exc:
            print(exc)
    cvt = ConvolutionalVisionTransformer(spec=data["MODEL"]['SPEC'], num_classes=args.num_classes)
    return cvt,data


def add_optimizer_params_lr(trainer, args):
    stages = ['stage0', 'stage1', 'stage2']
    patch_embed = 'patch_embed'
    blocks = {'stage0': [str(i) for i in range(1)], 'stage1': [str(i) for i in range(2)],
              'stage2': [str(i) for i in range(10)]}
    dictionary_lr = {}
    start_lr = args.initial_learning_rate

    for stage in stages:
        dictionary_lr[stage+"."+patch_embed] = start_lr
        if stage == 'stage2':
            dictionary_lr[stage+'.cls_token'] = start_lr
        for i, group in enumerate(blocks[stage]):
            dictionary_lr[stage+".blocks."+group] = start_lr
            if start_lr > args.min_lr:
                start_lr *= 1e-1

    for name, param in trainer.model.named_parameters():
        if name[:18] in dictionary_lr:
            print(f"Setting lr for {name} to {dictionary_lr[name[:18]]}")
            trainer.optimizer.add_param_group({'params': param, 'lr': dictionary_lr[name[:18]]})
        elif name[:15] in dictionary_lr:
            print(f"Setting lr for {name} to {dictionary_lr[name[:15]]}")
            trainer.optimizer.add_param_group({'params': param, 'lr': dictionary_lr[name[:15]]})
        elif name == 'stage2.cls_token':
            print(f"Setting lr for {name} to {dictionary_lr[name]}")
            trainer.optimizer.add_param_group({'params': param, 'lr': dictionary_lr[name]})


def build_optimizer_cvt(model, args):
    list_params = []
    for name, param in model.named_parameters():
        if 'head' in name or 'norm_final' in name:
            list_params.append(param)
    return optim.Adamax(list_params, lr=args.initial_learning_rate * args.clf_lr)


def train_cifar10(args):
    args.num_classes = 10
    args.update_epoch = 1
    args.stop_update_epoch = 5
    args.scale_lr = 10
    args.num_epochs=200
    args.initial_learning_rate = 0.002
    args.min_lr = 2e-6
    args.clf_lr = 1e-3
    args.update_per_epoch = 2
    args.stop_lr = 0.0021
    args.scheduler=True
    cvt,configs = build_cvt(args)
    train_loader, val_loader = get_train_val_loaders_cifar(val_size=0)
    test_loader = get_test_loader_cifar()
    trainer = TrainerCvt(cvt, train_loader, test_loader, add_optimizer_params_lr, args,build_optimizer_cvt,configs)
    trainer.train()


def train_cifar100(args):
    args.num_classes = 100
    args.update_epoch = 1
    args.stop_update_epoch = 5
    args.scale_lr = 10
    args.num_epochs = 200
    args.initial_learning_rate = 0.002
    args.min_lr = 2e-5
    args.clf_lr = 1e-2
    args.update_per_epoch = 2
    args.stop_lr = 0.0021
    args.scheduler = True
    cvt,configs = build_cvt(args)
    train_loader, val_loader = get_train_val_loaders_cifar(val_size=0, dataset=torchvision.datasets.CIFAR100)
    test_loader = get_test_loader_cifar(dataset=torchvision.datasets.CIFAR100)

    trainer = TrainerCvt(cvt, train_loader, test_loader, add_optimizer_params_lr, args,build_optimizer_cvt,configs)
    trainer.train()


def train_tiny_imagenet(args):
    args.num_classes = 200
    args.update_epoch = 1
    args.stop_update_epoch = 6
    args.scale_lr = 10
    args.num_epochs = 150
    args.initial_learning_rate = 0.002
    args.min_lr = 2e-7
    args.clf_lr = 1e-4
    args.update_per_epoch = 4
    args.stop_lr = 0.0021
    args.scheduler = True

    cvt, configs = build_cvt(args)
    train_loader, class_to_idx = get_train_loader_tiny_imagenet(path="./data")
    val_loader = get_valid_loader_tiny_imagenet(path="./data", class_to_idx=class_to_idx)
    trainer = TrainerCvt(cvt, train_loader, val_loader, add_optimizer_params_lr, args,
                               build_optimizer_cvt, configs)
    trainer.train()


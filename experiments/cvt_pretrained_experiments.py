import torch
import torchvision
from torch import optim

from data_handlers import get_train_val_loaders_cifar, get_test_loader_cifar, get_train_loader_tiny_imagenet, \
    get_valid_loader_tiny_imagenet, get_train_val_loaders_food101, get_test_loader_food101
from experiments.cvt_experiments import build_cvt
from train.cvt_train import TrainerCvt


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
            if start_lr >= args.min_lr:
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
    return optim.Adamax(list_params, lr=args.initial_learning_rate *args.clf_lr)


def train_cifar10(args):
    args.num_classes = 10
    args.update_epoch = 1
    args.stop_update_epoch = 10
    args.scale_lr = 10

    args.num_epochs = 25
    args.initial_learning_rate = 5e-4
    args.min_lr = 5e-5
    args.clf_lr = 1.
    args.update_per_epoch = 4
    args.stop_lr = 5e-3
    args.scheduler = False

    cvt,configs = build_cvt(args)
    state_dict = torch.load(f"./pretrained_models/CvT-13-224x224-IN-1k.pth")
    state_dict['norm_final.weight'] =state_dict['norm.weight']
    state_dict['norm_final.bias'] =state_dict['norm.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    state_dict['head.weight'] = cvt.state_dict()['head.weight']
    state_dict['head.bias'] = cvt.state_dict()['head.bias']
    cvt.load_state_dict(state_dict)

    train_loader, val_loader = get_train_val_loaders_cifar(val_size=0,resize=64)
    test_loader = get_test_loader_cifar(resize=64)
    trainer = TrainerCvt(cvt, train_loader, test_loader, add_optimizer_params_lr, args,build_optimizer_cvt,configs)
    return trainer.train()


def train_cifar100(args):
    args.num_classes = 100
    args.update_epoch = 1
    args.stop_update_epoch = 10
    args.scale_lr = 10

    args.num_epochs = 25
    args.initial_learning_rate = 5e-4
    args.min_lr = 5e-5
    args.clf_lr = 1.
    args.update_per_epoch = 4
    args.stop_lr = 5e-3
    args.scheduler = False

    cvt,configs = build_cvt(args)
    state_dict = torch.load(f"./pretrained_models/CvT-13-224x224-IN-1k.pth")
    state_dict['norm_final.weight'] = state_dict['norm.weight']
    state_dict['norm_final.bias'] = state_dict['norm.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    state_dict['head.weight'] = cvt.state_dict()['head.weight']
    state_dict['head.bias'] = cvt.state_dict()['head.bias']
    cvt.load_state_dict(state_dict)

    train_loader, val_loader = get_train_val_loaders_cifar(val_size=0, dataset=torchvision.datasets.CIFAR100,resize=64)
    test_loader = get_test_loader_cifar(dataset=torchvision.datasets.CIFAR100,resize=64)
    trainer = TrainerCvt(cvt, train_loader, test_loader, add_optimizer_params_lr, args,build_optimizer_cvt,configs)
    return trainer.train()

def train_tiny_imagenet(args):
    args.num_classes = 200
    args.update_epoch = 1
    args.stop_update_epoch = 5
    args.scale_lr = 10

    args.num_epochs = 25
    args.initial_learning_rate = 5e-4
    args.min_lr = 5e-9
    args.clf_lr = 1.
    args.update_per_epoch = 4
    args.stop_lr = 5e-3
    args.scheduler = False

    cvt, configs = build_cvt(args)
    train_loader, class_to_idx = get_train_loader_tiny_imagenet(path="./data")
    val_loader = get_valid_loader_tiny_imagenet(path="./data", class_to_idx=class_to_idx)
    state_dict = torch.load(f"./pretrained_models/CvT-13-224x224-IN-1k.pth")
    state_dict['norm_final.weight'] = state_dict['norm.weight']
    state_dict['norm_final.bias'] = state_dict['norm.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    state_dict['head.weight'] = cvt.state_dict()['head.weight']
    state_dict['head.bias'] = cvt.state_dict()['head.bias']
    cvt.load_state_dict(state_dict)
    trainer = TrainerCvt(cvt, train_loader, val_loader, add_optimizer_params_lr, args,
                               build_optimizer_cvt, configs)

    return trainer.train()
def train_food101(args):
    args.num_classes = 101
    args.update_epoch = 1
    args.stop_update_epoch = 10
    args.scale_lr = 10

    args.num_epochs = 25
    args.initial_learning_rate = 5e-4
    args.min_lr = 5e-5
    args.clf_lr = 1.
    args.update_per_epoch = 4
    args.stop_lr = 5e-3
    args.scheduler = False

    cvt,configs = build_cvt(args)
    state_dict = torch.load(f"./pretrained_models/CvT-13-224x224-IN-1k.pth")
    state_dict['norm_final.weight'] =state_dict['norm.weight']
    state_dict['norm_final.bias'] =state_dict['norm.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    state_dict['head.weight'] = cvt.state_dict()['head.weight']
    state_dict['head.bias'] = cvt.state_dict()['head.bias']
    cvt.load_state_dict(state_dict)

    train_loader = get_train_val_loaders_food101()
    val_loader = get_test_loader_food101()
    trainer = TrainerCvt(cvt, train_loader, val_loader, add_optimizer_params_lr, args,
                               build_optimizer_cvt, configs)
    trainer.train()
from experiments import r18_experiments, wrn_experiments, cvt_experiments, cvt_pretrained_experiments, bert_experiments

RUNS = {
    'resnet18': {
        'cifar10': r18_experiments.train_cifar10,
        'cifar100': r18_experiments.train_cifar100,
        'tinyimagenet': r18_experiments.train_tinyimagenet,
    },
    'wresnet': {
        'cifar10': wrn_experiments.train_cifar10,
        'cifar100': wrn_experiments.train_cifar100,
        'tinyimagenet': wrn_experiments.train_tiny_imagenet,
    },
    'cvt': {
        'cifar10': cvt_experiments.train_cifar10,
        'cifar100': cvt_experiments.train_cifar100,
        'tinyimagenet': cvt_experiments.train_tiny_imagenet,
    },
    'cvt_pretrained': {
        'cifar10': cvt_pretrained_experiments.train_cifar10,
        'cifar100': cvt_pretrained_experiments.train_cifar100,
        'tinyimagenet': cvt_pretrained_experiments.train_tiny_imagenet,
    },
    'bert': {
        'qnli': bert_experiments.train_qnli,
        'rte': bert_experiments.train_rte,
        'boolq': bert_experiments.train_boolq
    }
}

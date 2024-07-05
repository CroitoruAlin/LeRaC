# Learning Rate Curriculum (LeRaC) - IJCV 2024 (official repository)

### Introduction

This repository contains the code that implements the experiments described in
the work "Learning Rate Curriculum" accepted at the International Journal of Computer Vision. 

### Citation

Please cite our work if you use any material released in this repository.
```
@article{Croitoru-IJCV-2024,
  author    = {Croitoru, Florinel-Alin and Ristea, Nicolae-Catalin and Ionescu, Radu Tudor and Sebe, Nicu},
  title     = "{Learning Rate Curriculum}",
  journal = {International Journal of Computer Vision},
  year      = {2024},
  }
```

### Data
The data sets (CIFAR-10, CIFAR-100, Tiny ImageNet, Qnli, BoolQ, RTE) should
be stored in a directory called "data":
```sh
  | LeRac
    | data
      | CIFAR-10
      | CIFAR-100
      | tiny-imagenet-200
      | imagenet
      | QNLIv2
      | BoolQ
      | RTE
```

### Run
The experiments can be run via a command line, passing
the model and the data set as arguments. For example, the command to run the LeRac strategy on
Resnet-18 for CIFAR-10, is the following:
```sh
    python main.py --model_name resnet18 --dataset cifar10
```

Obs.: 
1) The process will save the models on disk in a directory called "saved_models".
Therefore, this directory should exist before running the experiments.

2) If the experiments involve a pre-trained architecture, the weights should be
stored in a directory called "pretrained_models". Example:
``` sh
  | LeRac
    | pretrained_models
      | CvT-13-224x224-IN-1k.pth  
```


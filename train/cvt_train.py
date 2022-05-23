import os

import torch
from timm.scheduler import create_scheduler

from test import test
from train.resnet_train import Trainer


class TrainerCvt(Trainer):
    def __init__(self, model, train_loader, val_loader, add_optimizer_params_lr, args,build_optimizer, config):
        super().__init__(model, train_loader, val_loader, add_optimizer_params_lr, args,build_optimizer)
        attr_dict = AttrDict()
        config["MODEL"]["TRAIN"]["LR_SCHEDULER"]["ARGS"]['epochs'] = args.num_epochs
        attr_dict.update(config["MODEL"]["TRAIN"]["LR_SCHEDULER"]["ARGS"])
        config["MODEL"]["TRAIN"]["LR_SCHEDULER"]["ARGS"] = attr_dict
        self.lr_scheduler = None
        self.config=config


    def create_lr_scheduler(self, cfg, optimizer, begin_epoch):
        if 'METHOD' not in cfg["TRAIN"]["LR_SCHEDULER"]:
            raise ValueError('Please set TRAIN.LR_SCHEDULER.METHOD!')
        elif cfg["TRAIN"]["LR_SCHEDULER"]["METHOD"] == 'MultiStep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                cfg["TRAIN"]["LR_SCHEDULER"]["MILESTONES"],
                cfg["TRAIN"]["LR_SCHEDULER"]["GAMMA"],
                begin_epoch - 1)
        elif cfg["TRAIN"]["LR_SCHEDULER"]["METHOD"] == 'CosineAnnealing':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                cfg["TRAIN"]["END_EPOCH"],
                cfg["TRAIN"]["LR_SCHEDULER"]["ETA_MIN"],
                begin_epoch - 1
            )
        elif cfg["TRAIN"]["LR_SCHEDULER"]["METHOD"] == 'CyclicLR':
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=cfg["TRAIN"]["LR_SCHEDULER"]["BASE_LR"],
                step_size_up=cfg["TRAIN"]["LR_SCHEDULER"]["STEP_SIZE_UP"]
            )
        elif cfg["TRAIN"]["LR_SCHEDULER"]["METHOD"] == 'timm':
            args = cfg["TRAIN"]["LR_SCHEDULER"]["ARGS"]
            args.epochs= args['epochs']
            lr_scheduler, _ = create_scheduler(args, optimizer)
            lr_scheduler.step(begin_epoch)
        else:
            raise ValueError('Unknown lr scheduler: {}'.format(
                cfg["TRAIN"]["LR_SCHEDULER"]["METHOD"]))

        return lr_scheduler

    def train(self):
        best_accuracy = 0.
        for epoch in range(1, self.args.num_epochs + 1):
            print('Epoch', epoch)
            super(TrainerCvt, self)._train_epoch(epoch)

            if(self.args.stop_update_epoch <= epoch and self.args.scheduler):
                if self.lr_scheduler is None:
                    self.lr_scheduler = self.create_lr_scheduler(self.config['MODEL'], self.optimizer,epoch-self.args.stop_update_epoch)
                else:
                    self.lr_scheduler.step(epoch-self.args.stop_update_epoch)

            test_acc = test(self.model, self.args.device, self.val_loader)
            if best_accuracy < test_acc:
                best_accuracy = test_acc
                torch.save(self.model.state_dict(), os.path.join( "saved_models", self.args.model_name + ".pth"))
                print('New best accuracy. Model Saved!')
            if epoch % self.args.update_epoch == 0 and epoch < self.args.stop_update_epoch:
                self.update_lr()

            self.print_lr()
        return best_accuracy


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self




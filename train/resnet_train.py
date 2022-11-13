import os
from time import time

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn.functional as F

from test import test


class Trainer:
    def __init__(self, model, train_loader, val_loader,add_optimizer_params_lr, args,
                 build_optimizer):
        self.args = args
        self.model = model
        self.scale_lr = args.scale_lr
        self.update_epoch = args.update_epoch
        self.optimizer = build_optimizer(model, args)
        add_optimizer_params_lr(self, args)
        self.ce_loss = F.cross_entropy

        self.model = self.model.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', min_lr=1e-5, patience=7)

    def train(self):
        best_accuracy = 0.
        for epoch in range(1, self.args.num_epochs + 1):
            print('Epoch', epoch)
            if epoch == self.args.decay_epoch and epoch < self.args.stop_decay_epoch:
                for param in self.optimizer.param_groups:
                    if param['lr']>1e-4:
                        param['lr'] = param['lr'] / 10
                print(f"Learning rate updated to {param['lr']}")
                self.args.decay_epoch += self.args.decay_step
            train_acc = self._train_epoch(epoch)
            # self.scheduler.step(train_acc)
            if epoch % 1 == 0:
                test_acc = test(self.model, self.args.device, self.val_loader)
                if best_accuracy < test_acc:
                    best_accuracy = test_acc
                    torch.save(self.model.state_dict(), os.path.join("saved_models", self.args.model_name + ".pth"))
                    print('New best accuracy. Model Saved!')
            if epoch < self.args.stop_update_epoch and epoch % self.update_epoch == 0:
                self.update_lr()
            self.print_lr()

    def _train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0.
        processed = 0.
        steps, length = 0, len(self.train_loader)
        update_points = []
        if self.args.update_per_epoch ==2:
            update_points.append(length//2)
        elif self.args.update_per_epoch==4:
            update_points = [length//2,length//4,3*length//4]
        for (data, target) in pbar:
            data, target = data.to(self.args.device), target.to(self.args.device)

            y_pred = self.model(data)
            loss = self.ce_loss(y_pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            steps += 1
            if epoch < self.args.stop_update_epoch and steps in update_points:
                self.update_lr()
            pbar.set_description(desc=f'Loss={loss.item()} Accuracy={100 * correct / processed:0.2f}')
        print(f'Loss={loss.item()} Accuracy={100 * correct / processed:0.2f}')
        return 100 * correct / processed

    def update_lr(self):
        for group in self.optimizer.param_groups:
            if group['lr'] * self.scale_lr < self.args.stop_lr:
                prev_value = group['lr']
                group['lr'] = group['lr'] * self.scale_lr
                print(f"Updated lr: {group['lr']}, previous value {prev_value}")

    def print_lr(self):
        lr = []
        for group in self.optimizer.param_groups:
            lr.append(group['lr'])
        print(f"Current learning rates:{set(lr)}")

class TrainerBaseline():
    def __init__(self, model, train_loader, val_loader, args,
                 build_optimizer):
        self.args = args
        self.model = model
        self.optimizer = build_optimizer(model, args)
        self.ce_loss = F.cross_entropy

        self.model = self.model.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self):
        best_accuracy = 0.
        for epoch in range(1, self.args.num_epochs + 1):
            print('Epoch', epoch)
            if epoch == self.args.decay_epoch and epoch < self.args.stop_decay_epoch:
                for param in self.optimizer.param_groups:
                    param['lr'] = param['lr'] / 10
                print(f"Learning rate updated to {param['lr']}")
                self.args.decay_epoch += self.args.decay_step
            start_time = time()
            self._train_epoch()
            end_time = time()
            print(end_time-start_time)
            if epoch % 5 == 0:
                test_acc = test(self.model, self.args.device, self.val_loader)
                if best_accuracy < test_acc:
                    best_accuracy = test_acc
                    torch.save(self.model.state_dict(), os.path.join("saved_models", self.args.model_name + ".pth"))
                    print('New best accuracy. Model Saved!')

    def _train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0.
        processed = 0.
        steps, length = 0, len(self.train_loader)
        self.print_lr()
        for (data, target) in pbar:

            data, target = data.to(self.args.device), target.to(self.args.device)

            y_pred = self.model(data)
            loss = self.ce_loss(y_pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            steps += 1
            pbar.set_description(desc=f'Loss={loss.item()} Accuracy={100 * correct / processed:0.2f}')
        print(f'Loss={loss.item()} Accuracy={100 * correct / processed:0.2f}')
        return 100 * correct / processed

    def print_lr(self):
        lr = []
        for group in self.optimizer.param_groups:
            lr.append(group['lr'])
        print(f"Current learning rates:{lr}")

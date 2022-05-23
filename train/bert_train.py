import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from test import test_bert


class TrainerBert:
    def __init__(self, model, train_loader, val_loader,add_optimizer_params_lr, args,
                 build_optimizer):
        self.args = args
        self.model = model
        self.optimizer = build_optimizer(model, args)
        add_optimizer_params_lr(self, args)
        self.ce_loss = F.cross_entropy

        self.model = self.model.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader


    def train(self):
        print('Training!')
        best_accuracy = 0.
        for epoch in range(1, self.args.num_epochs + 1):
            print('Epoch', epoch)
            self._train_epoch(epoch)
            test_acc = test_bert(self.model, self.args.device, self.val_loader)
            if best_accuracy < test_acc:
                best_accuracy = test_acc
                torch.save(self.model.state_dict(), os.path.join(".", "saved_models", self.args.model_name + ".pth"))
                print('New best accuracy. Model Saved!')
            self.update_lr()
            self.print_lr()
        return best_accuracy

    def _train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.args.device)
        pbar = tqdm(self.train_loader)
        correct = 0.
        processed = 0.
        step = 0
        length = len(self.train_loader)
        update_points = []
        if self.args.update_per_epoch == 2:
            update_points.append(length // 2)
        elif self.args.update_per_epoch == 4:
            update_points = [length // 2, length // 4, 3 * length // 4]
        print(f"Length train loader{length}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.args.device)
            attention_masks = batch['attention_masks'].to(self.args.device)
            labels = batch['labels'].to(self.args.device)
            token_type_ids = batch['token_type_ids'].to(self.args.device)

            outputs = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks)
            y_pred = outputs[0]
            loss = self.ce_loss(y_pred, labels)
            self.model.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            processed += len(labels)
            step += 1
            if epoch < self.args.stop_update_epoch and step in update_points:
                self.update_lr()
            pbar.set_description(desc=f'Loss={loss.item()} Accuracy={100 * correct / processed:0.2f}')
        print(f'Loss={loss.item()} Accuracy={100 * correct / processed:0.2f}')
        return 100 * correct / processed

    def update_lr(self):
        for group in self.optimizer.param_groups:
            if group['lr'] * self.args.scale_lr < self.args.stop_lr:
                previous_lr = group['lr']
                group['lr'] = group['lr'] * self.args.scale_lr
                print(f"Updated lr: {group['lr']}, previous value {previous_lr}")

    def print_lr(self):
        lr = []
        for group in self.optimizer.param_groups:
            lr.append(group['lr'])
        print(f"Current learning rates:{set(lr)}")



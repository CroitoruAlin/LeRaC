# from torch import optim
# # from transformers import BertForSequenceClassification
#
# from data_handlers import get_train_val_loaders_boolq, get_train_val_loaders_qnli, get_train_val_loaders_rte
# from train.bert_train import TrainerBert
#
#
# def add_optimizer_params_lr_scale(trainer, args):
#     start_lr = args.initial_learning_rate
#
#     for name, param in trainer.model.named_parameters():
#         if 'bert.pooler' not in name and 'classifier' not in name:
#             print(f"Setting lr for {name} to {start_lr}")
#             trainer.optimizer.add_param_group({'params': param, 'lr': start_lr})
#
#
# def build_optimizer_bert(model, args):
#     list_params = []
#     for name, params in model.named_parameters():
#         if 'bert.pooler' in name or 'classifier' in name:
#             list_params.append(params)
#     return optim.Adamax(list_params, lr=args.initial_learning_rate * args.clf_lr)
#
#
# def train_boolq(args):
#     args.scale_lr = 10
#     args.num_classes = 2
#     args.stop_update_epoch = 3
#     args.num_epochs = 10
#     args.initial_learning_rate = 5e-5
#     args.clf_lr = 1e-3
#     args.update_per_epoch = 4
#     args.stop_lr = 4.9e-4
#
#     model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2)
#     train_loader, val_loader = get_train_val_loaders_boolq(batch_size=10)
#     trainer = TrainerBert(model, train_loader, val_loader, add_optimizer_params_lr_scale, args, build_optimizer_bert)
#     return trainer.train()
#
#
# def train_qnli(args):
#     args.scale_lr = 10
#     args.num_classes = 2
#     args.stop_update_epoch = 3
#     args.num_epochs = 7
#     args.initial_learning_rate = 5e-5
#     args.clf_lr = 1e-3
#     args.update_per_epoch = 1
#     args.stop_lr = 4.9e-4
#     model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2)
#     train_loader, val_loader = get_train_val_loaders_qnli(batch_size=10)
#     trainer = TrainerBert(model, train_loader, val_loader, add_optimizer_params_lr_scale, args, build_optimizer_bert)
#     return trainer.train()
#
#
# def train_rte(args):
#     args.scale_lr = 10
#     args.num_classes = 2
#     args.stop_update_epoch = 3
#     args.num_epochs = 25
#     args.initial_learning_rate = 5e-5
#     args.clf_lr = 1e-3
#     args.update_per_epoch = 1
#     args.stop_lr = 4.9e-4
#     model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2)
#     train_loader, val_loader = get_train_val_loaders_rte(batch_size=10)
#     trainer = TrainerBert(model, train_loader, val_loader, add_optimizer_params_lr_scale, args, build_optimizer_bert)
#     return trainer.train()

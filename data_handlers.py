import csv
import os
from abc import ABC

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from bs4 import BeautifulSoup
from torch import randperm, default_generator
from torch.utils.data import Subset, DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.datasets.folder import default_loader, pil_loader, accimage_loader
from torchvision.transforms import transforms
import pandas as pd
# from transformers import AutoTokenizer


def get_train_val_loaders_boolq(batch_size=64):
    dataset_train = BoolQDataset(subset='train')
    dataset_val = BoolQDataset(subset='val')
    train_dataloader = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
    val_dataloader = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
    return train_dataloader, val_dataloader

def get_train_val_loaders_qnli(batch_size=10):
    dataset_train = EntailmentDataset(subset='train')
    dataset_val = EntailmentDataset(subset='val')
    train_dataloader = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
    val_dataloader = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
    return train_dataloader, val_dataloader

def get_train_val_loaders_rte(batch_size=10):
    dataset_train = EntailmentDataset(subset='train', path='./data/RTE/')
    dataset_val = EntailmentDataset(subset='val', path='./data/RTE/')
    train_dataloader = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
    val_dataloader = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
    return train_dataloader, val_dataloader

def get_train_val_loaders_cifar(val_size=2500, batch_size=64, dataset=torchvision.datasets.CIFAR10,resize=32):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )
    val_transforms = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    val_changes = transforms.Compose(val_transforms)
    train_data = dataset(root='./data', train=True, download=True, transform=transform)
    val_data = dataset(root='./data', train=True, download=True, transform=val_changes)
    torch.manual_seed(33)
    train_size = len(train_data) - val_size
    indices = randperm(sum([train_size, val_size]), generator=default_generator).tolist()
    train_ds = Subset(train_data, indices[0: train_size])
    val_ds = Subset(val_data, indices[train_size: train_size + val_size])
    print(f'Train data size {len(train_ds)}, Validation data size {len(val_ds)}')
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, None

def get_train_loader_imagenet_subset(batch_size=256):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        "./data/imagenet/ILSVRC/Data/CLS-LOC/train",
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            normalize,
        ]))
    class_to_idx = train_dataset.class_to_idx
    idx = [i for i in range(len(train_dataset)) if train_dataset.imgs[i][1] < 200]
    train_dataset = Subset(train_dataset, idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    return train_loader, class_to_idx


def get_train_loader_imagenet(batch_size=256):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        "/media/tonio/p2/research/datasets/imagenet/ILSVRC/Data/CLS-LOC/train",
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    return train_loader, train_dataset.class_to_idx

def get_val_loader_imagenet(class_to_idx, batch_size=256,subset=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transfs = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        normalize])
    val_dataset = ValidationImageNetDataset("./data/imagenet/ILSVRC/Data/CLS-LOC/val",
                                            "./data/imagenet/ILSVRC/Annotations/CLS-LOC/val",
                                            class_to_idx, transforms=transfs,subset=subset)


    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader


def get_test_loader_cifar(batch_size=64, dataset=torchvision.datasets.CIFAR10,resize=32):
    test_changes = transforms.Compose([transforms.Resize(resize),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))])
    test_data = dataset(root='./data', train=False, download=True, transform=test_changes)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader


def get_train_loader_tiny_imagenet(path, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )
    train_data = TinyImagenetTrain(
        os.path.join(path, 'tiny-imagenet-200', 'train'),
        transform=transform
    )
    return DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    ), train_data.class_to_idx


def get_valid_loader_tiny_imagenet(path, class_to_idx, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )
    train_data = TinyImagenetDatasetValidation(path, class_to_idx, transform=transform)
    return DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )


class TinyImagenetDatasetValidation(Dataset):
    def __init__(self, path, class_to_idx, transform=None):
        annotations = os.path.join(path, 'tiny-imagenet-200', 'val', "val_annotations.txt")
        val_annotations = pd.read_csv(annotations, sep='\t', lineterminator='\n', header=None,
                                      names=['file_name', 'id', 'ignore1', 'ignore2', 'ignore3', 'ignore4'],
                                      encoding='utf-8', quoting=csv.QUOTE_NONE)

        file_to_class = {}
        self.list_images = []
        for _, elem in val_annotations.iterrows():
            file_to_class[elem['file_name']] = class_to_idx[elem['id']]
            self.list_images.append(elem['file_name'])
        self.file_to_class = file_to_class
        self.path = os.path.join(path, 'tiny-imagenet-200', 'val', 'images')
        self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, item):
        file_name = os.path.join(self.path, self.list_images[item])
        image = default_loader(file_name)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.file_to_class[self.list_images[item]]


class TinyImagenetTrain(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # self.result={}
        self.images=[]
        for path,target in self.samples:
            sample = self.loader(path)
            self.images.append((sample,target))


    def __getitem__(self, item):
        image,target = self.images[item]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

class BoolQDataset(Dataset, ABC):
    def __init__(self, subset='train', max_length=100):
        if subset=='train':
            self.path = "./data/BoolQ/BoolQ/train.jsonl"
        elif subset =='val':
            self.path = "./data/BoolQ/BoolQ/val.jsonl"
        else:
            self.path = "./data/BoolQ/BoolQ/test.jsonl"
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
        self.data = pd.read_json(self.path,lines=True, orient='records')
        passages = self.data.passage.values
        questions = self.data.question.values
        self.answers = self.data.label.values.astype(np.int64)
        encodings = []
        for question, passage in zip(questions, passages):
            encoded_data = self.tokenizer.encode_plus(question, passage, max_length=max_length, padding='max_length'
                                                 , truncation=True)
            encodings.append(encoded_data)
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = torch.tensor(self.encodings[idx]['input_ids'])
        item['attention_masks'] = torch.tensor(self.encodings[idx]['attention_mask'])
        item['token_type_ids'] = torch.tensor(self.encodings[idx]['token_type_ids'])
        item['labels'] = torch.tensor(self.answers[idx])
        return item

    def __len__(self):
            return len(self.encodings)


class EntailmentDataset(Dataset, ABC):
        def __init__(self, subset='train', max_length=100, path="./data/QNLIv2/QNLI/"):
            dict_labels = {'entailment':0., 'not_entailment':1.}
            if subset == 'train':
                self.path = path + "train.tsv"
            elif subset == 'val':
                self.path = path + "dev.tsv"
            # self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
            self.data = pd.read_csv(self.path,sep='\t',lineterminator='\n', quoting=3)
            if 'QNLI' in path:
                sentences1 = self.data.question.values
                sentences2 = self.data.sentence.values
            else:
                sentences1 = self.data.sentence1.values
                sentences2 = self.data.sentence2.values
            labels = self.data.label.values
            self.labels = []
            dict_ignored_examples = {}
            count = 0
            for label in labels:
                count += 1
                if label in dict_labels:
                    self.labels.append(dict_labels[label])
                else:
                    dict_ignored_examples[count] = 1
            self.labels = np.array(self.labels, dtype=np.long)
            print(np.unique(self.labels))
            encodings = []
            count = 0
            for s1, s2 in zip(sentences1, sentences2):
                count += 1
                if count not in dict_ignored_examples:
                    encoded_data = self.tokenizer.encode_plus(s1, s2, max_length=max_length, padding='max_length',
                                                              truncation=True)
                    encodings.append(encoded_data)
            self.encodings = encodings

        def __getitem__(self, idx):
            item = {}
            item['input_ids'] = torch.tensor(self.encodings[idx]['input_ids'])
            item['attention_masks'] = torch.tensor(self.encodings[idx]['attention_mask'])
            item['token_type_ids'] = torch.tensor(self.encodings[idx]['token_type_ids'])
            item['labels'] = torch.tensor(self.labels[idx],dtype=torch.long)
            return item

        def __len__(self):
            return len(self.encodings)


class ValidationImageNetDataset(Dataset):
    def __init__(self, image_folder, annotations_folder, class_to_idx, transforms, subset=True):
        super(ValidationImageNetDataset, self).__init__()
        images = os.listdir(image_folder)
        self.image_folder = image_folder
        self.annotations_folder = annotations_folder
        self.labels = []
        self.full_paths = []
        for image in images:
            image_name = image.split('.')[0]
            annotations = os.path.join(annotations_folder, image_name+".xml")
            with open(annotations, 'r') as f:
                data = f.read()
            bs_data = BeautifulSoup(data, "xml")
            name_label = bs_data.find('name').contents[0]
            if subset and class_to_idx[name_label] >=200:
                continue
            self.labels.append(class_to_idx[name_label])
            self.full_paths.append(os.path.join(image_folder, image))
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.full_paths[index])
        if img.mode !='RGB':
            img=img.convert('RGB')
        if img is not None:
            img = self.transforms(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)



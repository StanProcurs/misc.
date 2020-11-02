import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision
from torchvision import transforms
from pytorchcv.model_provider import get_model

from PIL import Image
from tqdm import tqdm

os.makedirs('home/students/dipe038-2/stan/CNN_Weight_Noise', exist_ok=True)

  
train_dir = 'data/augmented_dataset_mixed'
train_img_list = list()
train_label_list = list()
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
for (dirpath, dirnames, filenames) in os.walk(train_dir):
    train_img_list += [os.path.join(dirpath, file) for file in filenames]
for img in train_img_list:
    for i, emotion in enumerate(emotions):
        if emotion in img:
            train_label_list.append(i)

valid_dir = 'data/val'
valid_img_list = list()
valid_label_list = list()
for (dirpath, dirnames, filenames) in os.walk(valid_dir):
    valid_img_list += [os.path.join(dirpath, file) for file in filenames]
for img in valid_img_list:
    for i, emotion in enumerate(emotions):
        if emotion in img:
            valid_label_list.append(i)


class EmotionDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.image = img_list
        self.label = label_list
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.image[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label[idx]

        return img, label

    def __len__(self):
        return len(self.image)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Load pretrained network as backbone
        pretrained = get_model('vgg19', pretrained=True)
        # remove last layer of fc
        self.backbone = pretrained.features
        self.output = pretrained.output
        self.classifier = nn.Linear(1000, 7)

        del pretrained

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.output(x)
        x = self.classifier(x)

        return x

    def freeze_backbone(self):
        """Freeze the backbone network weight"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Freeze the backbone network weight"""
        for p in self.backbone.parameters():
            p.requires_grad = True


def accuracy(prediction, ground_truth):
    num_correct = (np.array(prediction) == np.array(ground_truth)).sum()
    return num_correct / len(prediction)

if __name__ == '__main__':

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = EmotionDataset(
        train_img_list, train_label_list, transform=train_transform)
    valid_ds = EmotionDataset(
        valid_img_list, valid_label_list, transform=valid_transform)

    print(len(train_ds))

    EPOCHS = 1
    BATCH_SIZE = 64
    LR = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=None,
                        shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                        num_workers=4, pin_memory=True)
                            
    model = CNNModel().to(device)
    model.freeze_backbone()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, len(train_dl), T_mult=EPOCHS*len(train_dl))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()

        for img, label in tqdm(train_dl):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

        model.eval()

        predictions = []
        ground_truths = []

        for img, label in tqdm(valid_dl):
            img = img.to(device)
            with torch.no_grad():
                logits = model(img)
                prediction = torch.argmax(logits, dim=1)

                predictions.extend(prediction.tolist())
                ground_truths.extend(label.tolist())

        acc = accuracy(predictions, ground_truths)
        print(acc)

    EPOCHS = 40
    BATCH_SIZE = 64
    LR = 25e-6

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.unfreeze_backbone()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-7)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, len(train_dl), T_mult=len(train_dl)*EPOCHS)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for img, label in tqdm(train_dl):
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()

        predictions = []
        ground_truths = []
        for img, label in tqdm(valid_dl):
            img = img.to(device)
            with torch.no_grad():
                logits = model(img)
                prediction = torch.argmax(logits, dim=1)

                predictions.extend(prediction.tolist())
                ground_truths.extend(label.tolist())

        acc = accuracy(predictions, ground_truths)
        print(acc)

        if acc > 0.66:
            torch.save(model.state_dict(), '/home/students/dipe038-2/stan/CNN_Weight_Noise/weights_{}_acc_{}'.format(epoch, acc))

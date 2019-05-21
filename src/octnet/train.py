#usr/local/bin

import numpy as np
import math

import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torchvision.datasets.cifar as cifar
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard

from cnn import *

#param
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
batch_size = 128
num = 50
in_ch = 3
cat_num = 10
img_size = 32

prepro = transforms.Compose([
    transforms.CenterCrop((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

train_set = cifar.CIFAR10(root="../../data", train=True, transform=prepro, target_transform=None, download=True)
valid_set = cifar.CIFAR10(root="../../data", train=False, transform=prepro, target_transform=None, download=True)

train_data = DataLoader(train_set, batch_size, shuffle=False)
valid_data = DataLoader(valid_set, batch_size, shuffle=False)

net = SampleOct(in_ch, cat_num).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=.01, momentum=.9)

len_train = len(train_set)
len_valid = len(valid_set)

def train():
    losses, total, correct = 0, 0, 0

    for i, (x, y) in enumerate(train_data):
        x = x.to(device)
        y = y.to(device)

        pred_y = net(x)

        loss = criterion(pred_y, y)
        losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(pred_y.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    return losses / len_train, correct / total

def valid():
    with torch.no_grad():
        losses, total, correct = 0, 0, 0

        for i, (x, y) in enumerate(valid_data):
            x = x.to(device)
            y = y.to(device)

            pred_y = net(x)

            loss = criterion(pred_y, y)
            losses += loss.item()

            _, predicted = torch.max(pred_y.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        return losses / len_valid, correct / total

def test():
    confusion_matrix = torch.zeros(cat_num, cat_num)

    with torch.no_grad():
        for i, (x, y) in enumerate(valid_data):
            x = x.to(device)
            y = y.to(device)

            pred_y = net(x)
            _, preds = torch.max(pred_y, 1)

            for t, p in zip(y.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix.numpy()

### learning
writer = tensorboard.SummaryWriter() #おまじない

max_loss = math.inf
for epoch in range(num):
    train_loss, train_acc = train()
    valid_loss, valid_acc = valid()
    print(train_loss, valid_loss, train_acc, valid_acc)
    #tensorboard1
    writer.add_scalars(
        'data/loss',
        {'train_loss': train_loss, 'valid_loss': valid_loss},
        epoch + 1
    )

    #tesnsorboard2
    writer.add_scalars(
        'data/acc',
        {'train_acc': train_acc, 'valid_acc': valid_acc},
        epoch + 1
    )

    if max_loss > valid_acc:
        torch.save({
            'weight': net.state_dict(),
            'loss': valid_loss,
            'epoch': epoch}, 'save/model.pth')

writer.export_scalars_to_json('./all_scalars.json') #write json for tensorboard
writer.close()                                      #closing
print('finished learning')

# test
conmat = test()
print(conmat)

#usr/local/bin

import numpy as np
import math
import sys
import time
import os

import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.datasets.cifar as cifar
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tensorboardX as tensorboard

import octave

#param
os.makedirs('save/' + sys.argv[3], exist_ok=True)
alpha = float(sys.argv[4])
batch_size = int(sys.argv[1])
num = 200
learning_rate = .1
in_ch = 3
cat_num = 10
img_size = int(sys.argv[2])
conf_name = sys.argv[3] + '/' + sys.argv[4] + '_' + sys.argv[1] + '_' + sys.argv[2] + '_' + 'conf.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def param_decay(epoch):
    if epoch < 100:
        return learning_rate

    elif epoch < 150:
        return learning_rate * .5

    else:
        return learning_rate * .25


print(batch_size, img_size, num, conf_name, device, learning_rate)

trans = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomCrop((img_size, img_size), padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_trans = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_set = cifar.CIFAR10(root="./data", train=True, transform=trans, target_transform=None, download=True)
valid_set = cifar.CIFAR10(root="./data", train=False, transform=test_trans, target_transform=None, download=True)

train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
valid_data = DataLoader(valid_set, batch_size, shuffle=False, num_workers=2)

net = octave.octres50(alpha=alpha).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=.1, momentum=.9)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=param_decay)

len_train = len(train_set)
len_valid = len(valid_set)

def train():
    losses, total, correct = 0, 0, 0
    times = []

    for i, (x, y) in enumerate(train_data):
        x = x.to(device)
        y = y.to(device)

        start = time.time()
        pred_y = net(x)
        end = time.time()

        loss = criterion(pred_y, y)
        losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(pred_y.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
#        print(loss.item())
        times.append(end - start)

    return losses / len_train, correct / total, times

def valid():
    losses, total, correct = 0, 0, 0
    times = []

    with torch.no_grad():
        for i, (x, y) in enumerate(valid_data):
            x = x.to(device)
            y = y.to(device)

            start = time.time()
            pred_y = net(x)
            end = time.time()

            loss = criterion(pred_y, y)
            losses += loss.item()

            _, predicted = torch.max(pred_y.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            times.append(end - start)

    return losses / len_valid, correct / total, times

def test():
    confusion_matrix = torch.zeros(cat_num, cat_num)
    times = []

    with torch.no_grad():
        for i, (x, y) in enumerate(valid_data):
            x = x.to(device)
            y = y.to(device)

            start = time.time()
            pred_y = net(x)
            end = time.time()

            _, preds = torch.max(pred_y, 1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            times.append(end - start)

    return confusion_matrix.numpy(), times

### learning
writer = tensorboard.SummaryWriter() #おまじない

max_loss = math.inf
train_t, val_t = [], []

for epoch in range(num):
    scheduler.step()

    train_loss, train_acc, times1 = train()
    valid_loss, valid_acc, times2 = valid()

    print(train_loss, valid_loss, train_acc, valid_acc)

    train_t.extend(times1)
    val_t.extend(times2)

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
            'epoch': epoch}, 'save/' + sys.argv[3] + 'model.pth')

writer.export_scalars_to_json('save/' + sys.argv[3] + '/all_scalars.json') #write json for tensorboard
writer.close()                                      #closing
print('finished learning')

# test
weights, _, _ = torch.load('save/' + sys.argv[3] + '/model.pth')
net.load_state_dict(weights)

conmat, test_t = test()

#result write
np.savetxt('save/' + sys.argv[3] + '/train.txt', np.array(train_t), fmt='%s', delimiter=',')
np.savetxt('save/' + sys.argv[3] + '/val.txt', np.array(val_t), fmt='%s', delimiter=',')
np.savetxt('save/' + sys.argv[3] + '/test.txt', np.array(test_t), fmt='%s', delimiter=',')
np.savetxt('save/' + conf_name, np.array(conmat), fmt='%s', delimiter=',')
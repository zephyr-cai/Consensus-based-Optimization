import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import sys
import time

from utils import set_random_seed, write_res

class OneModule(nn.Module):
    def __init__(self):
        super(OneModule, self).__init__()

        self.fc1 = nn.Linear(28*28, 10)
        # self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output

def OneModule_loss(data, label, weight, bias):
    f = F.sigmoid(F.ReLU(weight.mm(data) + bias))
    loss_sum = -(torch.log(f) * label).sum()
    return loss

def OneModule_loss_batch(X, data, label, temp, reg, avg_choice, device):
    # print("data size: ", data.size())
    data = data.view(-1, 784).double()
    label_onehot = torch.zeros(label.size(0), 10).to(device).scatter(1, label.unsqueeze(1), 1).double()
    # print("data size: ", data.size())
    # print("label: ", label)
    # sys.exit("test!")
    weight = X[:, :7840]
    bias = X[:, 7840:]
    affine = torch.bmm(data.expand(weight.size(0), -1, 784), weight.view(weight.size(0), 784, 10))
    f = F.softmax(F.relu(affine + bias.unsqueeze(1)), dim=2)
    loss_sum = -(torch.log(f) * label_onehot).sum(2)
    loss = loss_sum.sum(1) / label.size(0)
    if avg_choice:
        coff = torch.exp(-1 * temp * loss)
        num = (coff.unsqueeze(1) * X).sum(0)
        dom = coff.sum().unsqueeze(0)
    else:
        index = torch.argmax(-loss)
        num = X[index].clone()
        dom = loss[index].clone()
    return num, dom

def TwoModule_loss_batch(X, data, label, temp, reg, avg_choice, device):
    data = data.view(-1, 784).double()
    weight1 = X[:, :784*50]
    bias1 = X[:, 784*50:784*50+50]
    weight2 = X[:, 784*50+50: 784*50+50+50*10]
    bias2 = X[:, 784*50+50+50*10:]
    label_onehot = torch.zeros(label.size(0), 10).to(device).scatter(1, label.unsqueeze(1), 1).double()
    affine1 = torch.bmm(data.expand(weight1.size(0), -1, 784), weight1.view(weight1.size(0), 784, 50))
    f1 = F.relu(affine1 + bias1.unsqueeze(1))
    affine2 = torch.bmm(f1, weight2.view(weight2.size(0), 50, 10))
    f2 = F.softmax(F.relu(affine2 + bias2.unsqueeze(1)), dim=2)
    loss_sum = -(torch.log(f2) * label_onehot).sum(2)
    loss = loss_sum.sum(1) / label.size(0)
    if avg_choice:
        coff = torch.exp(-1 * temp * loss)
        num = (coff.unsqueeze(1) * X).sum(0)
        dom = coff.sum().unsqueeze(0)
    else:
        index = torch.argmax(-loss)
        num = X[index].clone()
        dom = loss[index].clone()
    return num, dom

def OneModule_test(data, label, para, device):
    data = data.view(-1, 784).double()
    label_onehot = torch.zeros(label.size(0), 10).to(device).scatter(1, label.unsqueeze(1), 1).double()
    weight = para[:7840]
    bias = para[7840:]
    affine = data.mm(weight.view(784, 10))
    f = F.softmax(F.relu(affine + bias.unsqueeze(0)), dim=1)
    loss_sum = -(torch.log(f) * label_onehot).sum(1)
    loss = loss_sum.sum(0) / label.size(0)
    pred = torch.argmax(f, 1)
    # print(pred)
    # print(label)
    # sys.exit("test!")
    acc = (pred == label).sum().double() / label.size(0)
    return loss, acc

def TwoModule_test(data, label, para, device):
    data = data.view(-1, 784).double()
    label_onehot = torch.zeros(label.size(0), 10).to(device).scatter(1, label.unsqueeze(1), 1).double()
    weight1 = para[:784*50]
    bias1 = para[784*50:784*50+50]
    weight2 = para[784*50+50: 784*50+50+50*10]
    bias2 = para[784*50+50+50*10:]
    affine1 = data.mm(weight1.view(784, 50))
    f1 = F.relu(affine1 + bias1.unsqueeze(0))
    affine2 = f1.mm(weight2.view(50, 10))
    f2 = F.softmax(F.relu(affine2 + bias2.unsqueeze(0)), dim=1)
    loss_sum = -(torch.log(f2) * label_onehot).sum(1)
    loss = loss_sum.sum(0) / label.size(0)
    pred = torch.argmax(f2, 1)
    acc = (pred == label).sum().double() / label.size(0)
    return loss, acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST test for optimization algorithms')
    parser.add_argument('--epoch', '-e', default=200, type=int, help='number of epochs for testing')
    parser.add_argument('--batch-size', '-b', default=256, type=int, help='batch size for input data')
    parser.add_argument('--lr', '-l', default=0.01, type=float, help='learning rate')
    parser.add_argument('--gamma', '-g', default=0.7, type=float, help='learing rate step gamma')
    parser.add_argument('--seed', '-s', default=0, type=int, help='random seed')
    args = parser.parse_args()

    set_random_seed(args.seed)
    torch.set_num_threads(1)

    print("preparing data...")
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data_train = datasets.MNIST(
        root="./data_cache/",
        transform=transforms.ToTensor(),
        train=True,
        download=True
    )
    data_test = datasets.MNIST(
        root="./data_cache/",
        transform=transforms.ToTensor(),
        train=False
    )
    trainloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    model = OneModule()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    log_loss = list()
    log_acc = list()

    model.cuda()
    print("training...")
    time_start = time.time()
    for epoch in range(args.epoch):
        model.train()
        for train_batch, (data, label) in enumerate(trainloader):
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()
            pred = model(data)
            loss = F.nll_loss(pred, label)
            loss.backward()
            optimizer.step()
            if train_batch % 100 == 0 or (train_batch + 1) == len(trainloader):
                print("epoch: {:d} | iter: {:d} | loss: {:f}".format(epoch, train_batch, loss.item()))
        
        model.eval()
        acc_sum = 0
        loss_sum = 0
        with torch.no_grad():
            for val_batch, (data_val, label_val) in enumerate(testloader):
                data_val, label_val = data_val.cuda(), label_val.cuda()
                pred = model(data_val)
                loss = F.nll_loss(pred, label_val, reduction='sum')
                loss_sum += loss.item()
                pred_label = pred.argmax(dim=1, keepdim=True)
                acc_sum += (pred_label.eq(label_val.view_as(pred_label))).sum().double().item()
            loss_sum /= len(testloader.dataset)
            acc_sum /= len(testloader.dataset)
            log_loss.append(loss_sum)
            log_acc.append(acc_sum)
            print("epoch: {:d} | loss: {:f} | acc: {:f}".format(epoch, loss_sum, acc_sum))

    time_end = time.time()
    print("epoch time", time_end - time_start)
    sys.exit("time!")   
    write_res('./results_mnist/11_log_loss_SGD_new.npy', log_loss)
    write_res('./results_mnist/11_log_acc_SGD_new.npy', log_acc)
    print("Results written!")
        
            
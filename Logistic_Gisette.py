import os
import sys
import math
import torch
import numpy as np
import random

from utils import download_extract, sigmoid

# gisette dataset
url_gisette = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/'
url_Gisette = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/'
FILENAME_X = 'gisette_processed_x.npy'
FILENAME_Y = 'gisette_processed_y.npy'

def preprocess_gisette(cache_location):
    download_extract(url_gisette, cache_location, 'gisette_train.data')
    download_extract(url_gisette, cache_location, 'gisette_train.labels')
    download_extract(url_gisette, cache_location, 'gisette_valid.data')
    download_extract(url_Gisette, cache_location, 'gisette_valid.labels')

    data = []
    with open(os.path.join(cache_location, 'gisette_train.data')) as f:
        for line in f.readlines():
            data.append((line.strip().split(" ")))
    label = []
    with open(os.path.join(cache_location, 'gisette_train.labels')) as f:
        for line in f.readlines():
            label.append((line.strip().split(" ")))
    data_valid = []
    with open(os.path.join(cache_location, 'gisette_valid.data')) as f:
        for line in f.readlines():
            data_valid.append((line.strip().split(" ")))
    label_valid = []
    with open(os.path.join(cache_location, 'gisette_valid.labels')) as f:
        for line in f.readlines():
            label_valid.append((line.strip().split(" ")))
    data = np.array(data).astype(float)
    label = np.array(label).astype(int)
    data_valid = np.array(data_valid).astype(float)
    label_valid = np.array(label_valid).astype(int)
    # normalization
    data_max = max(np.max(data), np.max(data_valid))
    data_min = min(np.min(data), np.min(data_valid))
    data = (data - data_min) / (data_max - data_min)
    data_valid = (data_valid - data_min)/ (data_max - data_min)
    # data = torch.from_numpy(data).double()
    # label = torch.from_numpy(label).double()
    # data_valid = torch.from_numpy(data_valid).double()
    # label_valid = torch.from_numpy(label_valid).double()
    return data, label, data_valid, label_valid

def logistic_l0(data, label, u, v, lam=None):
    exponent = 0
    loss = 0
    for idx in range(data.size(0)):
        x = data[idx]
        y = label[idx]
        exponent = -1 * y * (torch.dot(x, u) + v)
        loss += torch.log(1 + torch.exp(exponent))
    reg = len(u[u != 0]) * lam
    loss = reg + loss / data.size(0)
    return loss

def batch_logistic_l0(X, data, label, temp, lam, avg_choice=None):
    u = X[:, :-1]
    v = X[:, -1:]
    U = u.unsqueeze(2)
    data_mul = data.expand(X.size(0), -1, -1)
    exponent = -1 * label.view(1, -1) * (torch.bmm(data_mul, U).squeeze(2) + v)
    # print("exp: ", exponent.size())
    reg = (u != 0).sum(1).double() * lam
    # print("reg: ", reg.size())
    loss = reg + torch.log(1 + torch.exp(exponent)).sum(1) / data.size(0)
    # print("loss: ", loss.size())
    coff = torch.exp(-1 * temp * loss)
    # print("coff: ", coff.size())
    num = (coff.view(X.size(0), 1) * X).sum(0)
    dom = coff.sum().unsqueeze(0)
    # print("num: ", num.size())
    # print("dom: ", dom.size())
    # sys.exit("Check!")
    return num, dom

def test_logistic_l0(data, label, u, v, lam):
    exponent = -1 * label * (data.mm(u.unsqueeze(1)) + v)
    reg = (u != 0).sum().double() * lam
    loss = reg + torch.log(1 + torch.exp(exponent)).sum() / data.size(0)
    return loss

def logistic_class(x, u, v):
    return torch.sign(x.dot(u) + v)

def batch_logistic_class(X, u, v):
    return torch.sign(X.mm(u.unsqueeze(1)) + v)
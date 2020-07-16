import numpy as np
import math
import torch
from utils import sigmoid

def generate_cs(m, n, sparse, seed):
    '''
    A: m * n,
    x: n * 1,
    b: m * 1.
    '''
    # np.random.seed(seed)
    A = np.random.standard_normal((m, n))
    norm = np.linalg.norm(A, axis=0)
    A = A / norm[np.newaxis, :]

    x = np.zeros((n, 1))
    idx = np.arange(n)
    # np.random.seed(seed)
    np.random.shuffle(idx)
    # np.random.seed(seed)
    temp = np.random.standard_normal((sparse, 1))
    x[idx[:sparse]] = (temp > 0).astype(float) - (temp <= 0).astype(float)

    # np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=math.sqrt(0.05), size=(m, 1))
    b = A.dot(x) + noise
    # A = torch.from_numpy(A)
    # sig = torch.from_numpy(sig)
    # b = torch.from_numpy(b)
    return A, x, b

def loss_cs(X, A, b, lam):
    norm = torch.norm(A.mm(X) - b)
    reg = (X != 0).sum().double() * lam
    loss = norm**2 / 2 + reg
    return loss

def batch_loss_cs(X, A, b, temp, lam, avg_choice):
    norm = torch.norm(A.mm(X.view(X.size(1), -1)) - b, dim=0).double()
    reg = (X != 0).sum(1).double() * lam
    loss = norm**2 / 2
    loss = torch.log(loss)
    '''
    if (loss >= 1000).any():
        loss /= 500
    elif (loss >= 500).any():
        loss /= 50
    elif (loss >= 100).any():
        loss /= 30
    elif (loss >= 10).any():
        loss /= 5
    '''
    if avg_choice:
        coff = torch.exp(-1 * temp * loss)
        num = (coff.view(X.size(0), 1) * X).sum(0)
        dom = coff.sum().unsqueeze(0)
    else:
        index = torch.argmax(-loss)
        num = X[index].clone()
        dom = loss[index].clone()
    return num, dom

def loss_cs_l1(X, A, b, lam):
    norm = torch.norm(A.mm(X.view(X.size(1), -1)) - b)
    reg = lam * torch.sum(torch.abs(X))
    loss = norm * norm / 2 + reg
    return loss

def acc_cs(X, sig):
    return torch.norm(X - sig) / torch.norm(sig)
import numpy as np
import torch
import math
import random

def rastrigin(x, B, C):
    return ((x - B)**2 - 10 * torch.cos(2 * math.pi * (x - B)) + 10).sum() / x.size(0) + C

def batch_rastrigin(x, B, C, temp, reg, avg_choice):
    loss = ((x - B)**2 - 10 * torch.cos(2 * math.pi * (x - B)) + 10).sum(1) / x.size(1) + C
    if avg_choice:
        coff = torch.exp(-1 * temp * loss)
        num = (coff.view(x.size(0), 1) * x).sum(0)
        dom = coff.sum().double().unsqueeze(0)
    else:
        index = torch.argmax(-loss)
        num = x[index].clone()
        dom = loss[index].clone()
    return num, dom

    
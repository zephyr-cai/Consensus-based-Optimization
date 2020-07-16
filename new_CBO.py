import random
import math
import numpy as np
import torch
import torch.distributed as dist
from multiprocessing import Pool
import sys
import copy

def avg(X, beta, pro):
    avg_list = torch.zeros(self.dim + 1).double()
    if self.batch_avg == self.num:
        self.X_update = self.X
    else:
        batch = random.sample(range(self.num), self.batch_avg)
        self.X_update = self.X[batch]
    avg_list[:self.dim], avg_list[-1:] = batch_loss_map[pro](self.X_update, data, label, self.temp, self.lam_reg, self.avg_choice)
    return avg_list

def CBO_1(X, V, lam, sigma, beta, timestep):

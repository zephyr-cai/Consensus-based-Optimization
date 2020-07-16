import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from multiprocessing import Pool
import sys
import copy


from Logistic_Gisette import logistic_l0, batch_logistic_l0
from Compressive_Sensing import loss_cs, batch_loss_cs
from Image_Angles import loss_recons, batch_loss_recons
from utils import set_random_seed, synchronize
from Neural_Network import OneModule_loss_batch, TwoModule_loss_batch
from Rastrigin import batch_rastrigin

loss_map = {
    'logistic': logistic_l0,
    'compressive_sensing': loss_cs,
    'angle_opt': loss_recons
}

batch_loss_map = {
    'logistic': batch_logistic_l0,
    'compressive_sensing': batch_loss_cs,
    'angle_opt': batch_loss_recons,
    'one_module': OneModule_loss_batch,
    'two_module': TwoModule_loss_batch,
    'rastrigin': batch_rastrigin
}

class CBO_optimizer:
    def __init__(self, num, dim, drift, noise, temp, timestep, tol, seed, batch_avg, avg_choice, noise_choice, device, lam_reg=None, batch_loss=None):
        self.num = num
        self.dim = dim

        self.drift = drift
        self.noise = math.sqrt(noise)
        self.temp = temp
        self.timestep = timestep

        self.tol = tol
        self.seed = seed
        self.avg_choice = avg_choice
        self.noise_choice = noise_choice

        self.batch_avg = batch_avg
        self.batch_loss = batch_loss
        self.lam_reg = lam_reg

        self.rank = 0
        self.device = device
        self._initialize_particles()

    def _initialize_particles(self, seed=None):
        if seed == None:
            set_random_seed(self.seed)
        else:
            set_random_seed(self.seed)
        if self.device == 'cuda':
            self.X = torch.cuda.FloatTensor(self.num, self.dim).normal_().double()
        else:
            self.X = torch.randn(self.num, self.dim).double()
        self.X_update = range(self.batch_avg)
    
    def _uniform_initialization_particles(self, seed=None):
        # set_random_seed(self.seed)
        self.X = torch.Tensor(self.num, self.dim).uniform_(-3.0, 3.0).double()
        self.X_update = range(self.batch_avg)
    
    def _custom_initialization(self, X, noi_std=None, seed=None):
        if seed == None:
            set_random_seed(self.seed)
        else:
            set_random_seed(seed)
        X_0 = X.squeeze(1)
        if noi_std is not None:
            std = torch.tensor([noi_std for i in range(self.dim * self.num)]).double()
            self.X = X_0.expand(self.num, -1) + torch.normal(mean=0.0, std=std).view(self.num, self.dim)
        else:
            self.X = X_0.expand(self.num, -1)
    
    def _kaiming_uniform_initialization(self):
        set_random_seed(self.seed)
        non_linearity_gain = math.sqrt(2.0)
        fan_1 = 784
        fan_2 = 50
        bound_1_bias = 1 / math.sqrt(fan_1)
        bound_2_bias = 1 / math.sqrt(fan_2)
        std_1 = non_linearity_gain * bound_1_bias
        std_2 = non_linearity_gain * bound_2_bias
        bound_1_weight = math.sqrt(3.0) * std_1
        bound_2_weight = math.sqrt(3.0) * std_2
        self.X[:, :784*50].uniform_(-bound_1_weight, bound_1_weight)
        self.X[:, 784*50:784*50+50].uniform_(-bound_1_bias, bound_1_bias)
        self.X[:, 784*50+50:784*50+50+50*10].uniform_(-bound_2_weight, bound_2_weight)
        self.X[:, 784*50+50+50*10:].uniform_(-bound_2_bias, bound_2_bias)
    
    def _kaiming_1_uniform_initialization(self):
        set_random_seed(self.seed)
        non_linearity_gain = math.sqrt(2.0)
        fan_1 = 784
        bound_1_bias = 1 / math.sqrt(fan_1)
        std_1 = non_linearity_gain * bound_1_bias
        bound_1_weight = math.sqrt(3.0) * std_1
        self.X[:, :784*10].uniform_(-bound_1_weight, bound_1_weight)
        self.X[:, 784*10:784*10+10].uniform_(-bound_1_bias, bound_1_bias)

    def _normalize_particles(self):
        par_max = torch.max(self.X, dim=1)[0]
        par_min = torch.min(self.X, dim=1)[0]
        self.X = (self.X - par_min.unsqueeze(1)) / (par_max.unsqueeze(1) - par_min.unsqueeze(1))
        self.X, _ = torch.sort(self.X, dim=1)

    def avg(self, data, label, pro):
        avg_list = torch.zeros(self.dim + 1).double()
        if self.batch_avg == self.num:
            self.X_update = self.X
        else:
            batch = random.sample(range(self.num), self.batch_avg)
            self.X_update = self.X[batch]
        # print(self.X.device)
        # print(self.X_update.device)
        if pro == 'one_module' or pro == 'two_module':
            avg_list[:self.dim], avg_list[-1:] = batch_loss_map[pro](self.X_update, data, label, self.temp, self.lam_reg, self.avg_choice, self.device)
        else:
            avg_list[:self.dim], avg_list[-1:] = batch_loss_map[pro](self.X_update, data, label, self.temp, self.lam_reg, self.avg_choice)
        return avg_list
    
    def forward(self, data, label, pro, former_avg):
        avg_list = self.avg(data, label, pro)
        if self.avg_choice:
            weighted_avg = avg_list[:self.dim] / avg_list[-1:]
        else:
            weighted_avg = avg_list[:self.dim]
        weighted_avg = weighted_avg.to(self.device)
        norm = torch.norm(weighted_avg - former_avg)
        M = weighted_avg.size(0)
        flag = bool(norm**2 / M <= self.tol)
        weighted_avg_list = weighted_avg.expand(self.num, -1)
        self.X = weighted_avg_list + (self.X - weighted_avg_list) * math.exp(-self.drift * self.timestep)
        if self.noise_choice:
            if self.device == 'cuda':
                noise_term = torch.cuda.FloatTensor(self.num, self.dim).normal_().double()
            else:
                noise_term = torch.randn(self.num, self.dim).double()
            # self.X += self.noise * math.sqrt(self.timestep) * (self.X - weighted_avg_list).mul(noise_term)
            self.X += self.noise * math.sqrt(self.timestep) * noise_term
        flag_noise = False
        # flag = False
        if flag:
            if self.device == 'cuda':
                bm = torch.cuda.FloatTensor(self.num, self.dim).normal_().double()
            else:
                bm = torch.randn(self.num, self.dim).double()
            self.X += self.noise * math.sqrt(self.timestep) * bm
            flag_noise = True
            if self.rank == 0:
                print("noise!")
        return weighted_avg, flag_noise


class MultiprocessCBO:
    def __init__(self, num_process, rank, num, dim, drift, noise, temp, timestep, tol, seed, batch_avg, avg_choice, noise_choice, device, lam_reg=None, batch_loss=None):
        self.num = num
        self.dim = dim

        self.drift = drift
        self.noise = math.sqrt(noise)
        self.temp = temp
        self.timestep = timestep

        self.tol = tol
        self.seed = seed
        self.avg_choice = avg_choice
        self.noise_choice = noise_choice

        self.batch_avg = batch_avg
        self.batch_loss = batch_loss
        self.lam_reg = lam_reg

        self.rank = rank
        self.num_process = num_process
        self.device = device
        self._initialize_particles()

    def _initialize_particles(self, seed=None):
        if seed == None:
            set_random_seed(self.seed)
        else:
            set_random_seed(seed)
        self.X = torch.randn(self.num * self.num_process, self.dim)[self.rank * self.num : self.rank * self.num + self.num, ].double()
        self.X_update = range(self.batch_avg)
    
    def _uniform_initialization_particles(self, seed=None):
        if seed == None:
            set_random_seed(self.seed)
        else:
            set_random_seed(seed)
        self.X = torch.Tensor(self.num * self.num_process, self.dim).uniform_(-3.0, 3.0)[self.rank * self.num : self.rank * self.num + self.num, ].double()
        self.X_update = range(self.batch_avg)
    
    def _kaiming_uniform_initialization(self):
        set_random_seed(self.seed)
        non_linearity_gain = math.sqrt(2.0)
        fan_1 = 784
        fan_2 = 500
        bound_1_bias = 1 / math.sqrt(fan_1)
        bound_2_bias = 1 / math.sqrt(fan_2)
        std_1 = non_linearity_gain * bound_1_bias
        std_2 = non_linearity_gain * bound_2_bias
        bound_1_weight = math.sqrt(3.0) * std_1
        bound_2_weight = math.sqrt(3.0) * std_2
        self.X[:, :784*500].uniform_(-bound_1_weight, bound_1_weight)
        self.X[:, 784*500:784*500+500].uniform_(-bound_1_bias, bound_1_bias)
        self.X[:, 784*500+500:784*500+500+500*10].uniform_(-bound_2_weight, bound_2_weight)
        self.X[:, 784*500+500+500*10:].uniform_(-bound_2_bias, bound_2_bias)
    
    def _custom_initialization(self, X_0, noi_std=None, seed=None):
        if seed == None:
            set_random_seed(self.seed)
        else:
            set_random_seed(seed)
        X = X_0.squeeze(1)
        if noi_std is not None:
            std = torch.tensor([noi_std for i in range(self.dim * self.num * self.num_process)]).double()
            self.X = X.expand(self.num * self.num_process, -1) + torch.normal(mean=0.0, std=std).view(self.num * self.num_process, self.dim)
        else:
            self.X = X.expand(self.num * self.num_process, -1)
        self.X = self.X[self.rank * self.num : self.rank * self.num + self.num, ].double()
    
    def _normalize_particles(self):
        par_max = torch.max(self.X)
        par_min = torch.min(self.X)
        self.X = (self.X - par_min) / (par_max - par_min)
        self.X, _ = torch.sort(self.X, dim=1)
    
    def avg(self, data, label, pro):
        avg_list = torch.zeros(self.dim + 1).double()
        if self.batch_avg == self.num:
            self.X_update = self.X
        else:
            batch = random.sample(range(self.num), self.batch_avg)
            self.X_update = self.X[batch]
        avg_list[:self.dim], avg_list[-1:] = batch_loss_map[pro](self.X_update, data, label, self.temp, self.lam_reg, self.avg_choice)
        return avg_list
    
    def forward(self, data, label, pro, former_avg):
        avg_list = self.avg(data, label, pro)
        synchronize(self.num_process)
        if self.avg_choice:
            dist.all_reduce(avg_list, op=dist.ReduceOp.SUM)
            weighted_avg = avg_list[:self.dim] / avg_list[-1:]
        else:
            loss_value = avg_list[-1:].clone()
            dist.all_reduce(loss_value, op=dist.ReduceOp.MIN)
            if loss_value != avg_list[-1:]:
                avg_list[:self.dim] = torch.zeros(self.dim).double()
                avg_list[-1:] = 0.0
            else:
                avg_list[-1:] = 1.0
            synchronize(self.num_process)
            dist.all_reduce(avg_list, op=dist.ReduceOp.SUM)
            weighted_avg = avg_list[:self.dim] / avg_list[-1:]
        norm = torch.norm(weighted_avg - former_avg)
        M = weighted_avg.size(0)
        flag = bool(norm**2 / M <= self.tol)
        weighted_avg_list = weighted_avg.expand(self.num, -1)
        self.X = weighted_avg_list + (self.X - weighted_avg_list) * math.exp(-self.drift * self.timestep)
        if self.noise_choice:
            noise_term = torch.randn(self.num * self.num_process, self.dim)[self.rank * self.num : self.rank * self.num + self.num, ].double()
            # self.X += self.noise * math.sqrt(self.timestep) * (self.X - weighted_avg_list).mul(noise_term)
            bm = torch.randn(self.num * self.num_process, self.dim)[self.rank * self.num : self.rank * self.num + self.num, ].double()
            self.X += self.noise * math.sqrt(self.timestep) * bm
        flag_noise = False
        if flag:
            bm = torch.randn(self.num * self.num_process, self.dim)[self.rank * self.num : self.rank * self.num + self.num, ].double()
            self.X += self.noise * math.sqrt(self.timestep) * bm
            flag_noise = True
            if self.rank == 0:
                print("noise!")
        return weighted_avg, flag_noise


######################### not used ########################
'''
def avg(X, data, label, batch_avg, batch_loss, pro, tmp, lam):
    batch = random.sample(range(X.shape[0]), batch_avg)
    num = np.array([0.0 for i in range(X.shape[1])])
    dom = 0
    for rank, idx in enumerate(batch):
        if pro == "logistic":
            loss = loss_map[pro](data, label, X[idx, :-1], X[idx, -1:], lam, batch_loss)
        elif pro == "compressive_sensing":
            loss = loss_map[pro](X[idx].reshape(X.shape[1], 1), data, label, lam)
        # if rank == 0:
        #     print("loss: ", loss)
        num += X[idx] * math.exp(-1 * tmp * loss)
        dom += math.exp(-1 * tmp * loss)
        # print("num: ", num)
        # print("dom: ", dom)
    weighted_avg = num / dom
    # print(weighted_avg)
    # os._exit()
    return weighted_avg

def consensus(X, data, label, pro, batch_avg, dri, noi, tmp, lam, h, tol, former_avg):
    batch_loss = None
    weighted_avg = avg(X, data, label, batch_avg, batch_loss, pro, tmp, lam)
    if pro == "compressive_sensing":
        norm = np.linalg.norm(weighted_avg.reshape((X.shape[1], 1)) - former_avg)
    else:
        norm = np.linalg.norm(weighted_avg - former_avg)
    M = max(1, np.linalg.norm(weighted_avg))
    # print("former: ", np.max(former_avg), np.min(former_avg), np.linalg.norm(former_avg))
    # print("now: ", np.max(weighted_avg), np.min(weighted_avg), np.linalg.norm(weighted_avg))
    # print("relative error: ", norm)
    flag = bool(norm/M <= tol)
    for idx in range(X.shape[0]):
        X[idx] = weighted_avg + (X[idx] - weighted_avg) * math.exp(-dri * h)
        if flag:
            noise = np.random.normal(loc=0.0, scale=math.sqrt(noi), size=X.shape[1])
            for d in range(X[idx].shape[0]):
                X[idx][d] += noi * math.sqrt(h) * (X[idx][d] - weighted_avg[d]) * (noise[d])
            if idx == 0:
                print("noise!")
    if pro == "compressive_sensing":
        weighted_avg = weighted_avg.reshape((X.shape[1], 1))
    return X, weighted_avg
'''
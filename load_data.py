import math
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from Logistic_Gisette import preprocess_gisette
from Compressive_Sensing import generate_cs, acc_cs, loss_cs
from Image_Angles import transform_image
from FISTA import FISTA

class Data(Dataset):
    def __init__(self, dataset, times, seed, sig_dim, sparsity, rank, reg_fista, iter_fista, tol_fista):
        super(Data, self).__init__()

        self.dataset = dataset
        self.times = times
        self.seed = seed
        self.sig_dim = sig_dim
        self.sparsity = sparsity
        self.rank = rank
        self.reg_fista = reg_fista
        self.iter_fista = iter_fista
        self.tol_fista = tol_fista

        self.n_dim = 64
        self.pic_path = "./data_cache/pic/Lenna.jpg"

        if self.dataset == "gisette":
            if not os.path.exists("./data_cache/logistic_gisette_data.npy"):
                if self.rank == 0:
                    print("writing data_npy...")
                self.cache_location = "./data_cache/gisette/"
                self.data, self.label, self.data_valid, self.label_valid = preprocess_gisette(self.cache_location)
                if self.rank == 0:
                    np.save("./data_cache/logistic_gisette_data.npy", self.data)
                    np.save("./data_cache/logistic_gisette_label.npy", self.label)
                    np.save("./data_cache/logistic_gisette_data_valid.npy", self.data_valid)
                    np.save("./data_cache/logistic_gisette_label_valid.npy", self.label_valid)
                    print("data_npy written!")
            else:
                if self.rank == 0:
                    print("reading data_npy...")
                self.data = np.load("./data_cache/logistic_gisette_data.npy")
                self.label = np.load("./data_cache/logistic_gisette_label.npy")
                self.data_valid = np.load("./data_cache/logistic_gisette_data_valid.npy")
                self.label_valid = np.load("./data_cache/logistic_gisette_label_valid.npy")
            self.data = torch.from_numpy(self.data).double()
            self.label = torch.from_numpy(self.label).double()
            self.data_valid = torch.from_numpy(self.data_valid).double()
            self.label_valid = torch.from_numpy(self.label_valid).double()
            self.feature_dim = self.data.size(1)
        elif self.dataset == "compressive_sensing":
            if (self.sig_dim == None) or (self.sparsity == None):
                self.sig_dim = 8000
                self.sparsity = 1
            cs_data_path = "./data_cache/compressive_sensing_data_" + str(self.sig_dim) + str(self.sparsity) + ".npy"
            if not os.path.exists(cs_data_path):
                if self.rank == 0:
                    print("preparing data...")
                m = 3000
                s = math.floor(self.sparsity * self.sig_dim / 100)
                self.data = []
                self.label = []
                self.observe = []
                self.initial = []
                self.L = []
                for a in range(self.times):
                    A, sig, b = generate_cs(m, sig_dim, s, self.seed)
                    # FISTA for starting pt (from l1 regularized problem)
                    X_0 = A.T.dot(b)
                    L = np.linalg.norm(A.T.dot(A))
                    X_0 = FISTA(X_0, A, b, L, self.reg_fista, self.iter_fista, self.tol_fista)
                    # loss = loss_cs(X_0.view(X_0.size(1), -1), A, b, args['reg'])
                    # acc = acc_cs(X_0.view(X_0.size(1), -1), sig)
                    print("data_idx: {:d} | L: {:f}".format(a, L))
                    self.data.append(A)
                    self.label.append(sig)
                    self.observe.append(b)
                    self.initial.append(X_0)
                    self.L.append(L)
                self.data = np.array(self.data).astype(float)
                self.label = np.array(self.label).astype(float)
                self.observe = np.array(self.observe).astype(float)
                self.initial = np.array(self.initial).astype(float)
                self.L = np.array(self.L).astype(float)
                if rank == 0:
                    np.save(cs_data_path, self.data)
                    np.save("./data_cache/compressive_sensing_label_" + str(self.sig_dim) + str(self.sparsity) + ".npy", self.label)
                    np.save("./data_cache/compressive_sensing_observe_" + str(self.sig_dim) + str(self.sparsity) + ".npy", self.observe)
                    np.save("./data_cache/compressive_sensing_initial_" + str(self.sig_dim) + str(self.sparsity) + ".npy", self.initial)
                    np.save("./data_cache/compressive_sensing_L_" + str(self.sig_dim) + str(self.sparsity) + ".npy", self.L)
                    print("data written!")
                self.data = torch.from_numpy(self.data).double()
                self.label = torch.from_numpy(self.label).double()
                self.observe = torch.from_numpy(self.observe).double()
                self.initial = torch.from_numpy(self.initial).double()
                self.L = torch.from_numpy(self.L).double()
            else:
                if self.rank == 0:
                    print("loading data...")
                self.data = np.load(cs_data_path)
                self.label = np.load("./data_cache/compressive_sensing_label_" + str(self.sig_dim) + str(self.sparsity) + ".npy")
                self.observe = np.load("./data_cache/compressive_sensing_observe_" + str(self.sig_dim) + str(self.sparsity) + ".npy")
                self.initial = np.load("./data_cache/compressive_sensing_initial_" + str(self.sig_dim) + str(self.sparsity) + ".npy")
                self.L = np.load("./data_cache/compressive_sensing_L_" + str(self.sig_dim) + str(self.sparsity) + ".npy")
                self.data = torch.from_numpy(self.data).double()
                self.label = torch.from_numpy(self.label).double()
                self.observe = torch.from_numpy(self.observe).double()
                self.initial = torch.from_numpy(self.initial).double()
                self.L = torch.from_numpy(self.L).double()
        elif self.dataset == "picture_reconstruction":
            if not os.path.exists("./data_cache/picture_reconstruction_data.npy"):
                if rank == 0:
                    print("writing data_npy...")
                self.data, self.label, self.theta = transform_image(self.pic_path, self.n_dim)
                if rank == 0:
                    np.save("./data_cache/picture_reconstruction_data.npy", self.data)
                    np.save("./data_cache/picture_reconstruction_label.npy", self.label)
                    np.save("./data_cache/picture_reconstruction_theta.npy", self.theta)
                    print("data written!")
                self.data = torch.from_numpy(self.data).double()
                self.label = torch.from_numpy(self.label).double()
                self.theta = torch.from_numpy(self.theta).double()
            else:
                if rank == 0:
                    print("loading data...")
                self.data = np.load("./data_cache/picture_reconstruction_data.npy")
                self.label = np.load("./data_cache/picture_reconstruction_label.npy")
                self.theta = np.load("./data_cache/picture_reconstruction_theta.npy")
                self.data = torch.from_numpy(self.data).double()
                self.label = torch.from_numpy(self.label).double()
                self.theta = torch.from_numpy(self.theta).double()
            if rank == 0:
                print("data size: ", self.data.size())
                print("label size: ", self.label.size())
                print("theta size: ", self.theta.size())

    def __getitem__(self, index):
        return self.data[index, :], self.label[index, :]
    
    def __len__(self):
        return self.data.size(0)

def batch_loader(batch):
    data = batch[0]
    label = batch[1]
    return data, label

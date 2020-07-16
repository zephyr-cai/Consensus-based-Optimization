import cv2
import math
import numpy as np
import torch
from skimage.transform import radon, rescale
from utils import load_image_gray
import sys

def generate_theta(n_dim):
    theta = torch.randn(n_dim * 4).double()
    max_theta = torch.max(theta)
    min_theta = torch.min(theta)
    theta = (theta - min_theta) / (max_theta - min_theta)
    theta = theta * 179.0
    theta, _ = torch.sort(theta)
    return theta.numpy()

def transform_image(filepath, n_dim):
    original_img = load_image_gray(filepath)
    scale = n_dim / original_img.shape[0]
    label = rescale(original_img, scale=scale, mode='reflect', multichannel=False)
    theta = generate_theta(n_dim)
    data = radon(label, theta=theta, circle=False)
    # data = torch.from_numpy(data)
    # label = torch.from_numpy(label)
    # theta = torch.from_numpy(theta)
    return data, label, theta
    
def loss_recons(X, data, label):
    theta = X.numpy()
    label_np = label.numpy()
    data_new = radon(label_np, theta=theta, circle=False)
    data_new = torch.from_numpy(data_new)
    loss = torch.norm(data_new - data, dim=0, keepdim=True).sum()
    return loss / 1000

def batch_loss_recons(X, data, label, temp, reg, avg_choice):
    loss_list = torch.zeros(X.size(0)).double()
    for i in range(X.size(0)):
        loss = loss_recons(X[i], data, label)
        loss_list[i] = loss
    if avg_choice:
        coff = torch.exp(-1 * temp * loss_list)
        num = (coff.view(X.size(0), 1) * X).sum(0)
        dom = coff.sum().unsqueeze(0)
    else:
        index = torch.argmax(-loss_list)
        num = X[index].clone()
        dom = loss_list[index].clone()
    return num, dom

def theta_acc(theta, est):
    return torch.norm(theta - est)
import math
import numpy as np
import torch
import random
import copy
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import sys
import time

from utils import var, softmax, launch_a_process, synchronize, setup, set_random_seed, write_res
from Logistic_Gisette import logistic_class, logistic_l0, batch_logistic_class, test_logistic_l0
from Compressive_Sensing import acc_cs, loss_cs
from Image_Angles import loss_recons, theta_acc
from Neural_Network import OneModule_test, TwoModule_test
from Rastrigin import rastrigin
from load_data import Data, batch_loader
from FISTA import FISTA
from CBO import CBO_optimizer, MultiprocessCBO


def main(rank, args):
    set_random_seed(args['seed'])
    torch.set_num_threads(1)

    device = 'cuda' if (torch.cuda.is_available() and args['gpu']) else 'cpu'

    if rank == 0:
        print("Preparing data...")
    if args['dataset'] == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        data_train = datasets.MNIST(
            root="./data_cache/",
            transform=transform,
            train=True,
            download=True
        )
        data_test = datasets.MNIST(
            root="./data_cache/",
            transform=transform,
            train=False
        )
    else:
        dataset = Data(args['dataset'], args['repeat_time'], args['seed'], args['n_dim'], args['sparsity'], rank, args['reg_fista'], args['iteration'], args['tolerance_fista'])
    res_loss = torch.zeros(args['repeat_time']).double()
    res_acc = torch.zeros(args['repeat_time']).double()
    log_loss = list()
    log_loss_ave = list()
    log_acc = list()
    log_loss_train = list()
    log_loss_ave_train = list()
    log_acc_train = list()

    '''
        gisette: sparse logistic regression on the Gisette datasets,
        picture_reconstruction: optimizing the angles of the Radon Transformation of the picture of Lenna,
        compressive_sensing: the same numerical experiment as the one in New PIHT paper by Zhang Xiaoqun,
        mnist: numerical experiments with optimizing neural networks on the MNIST dataset,
        ras: numerical experiments with rastrigin function optimization problem
    '''
    if args['dataset'] == "gisette":
        # preparing data...
        data_valid, label_valid = dataset.data_valid, dataset.label_valid
        data_valid, label_valid = data_valid.to(device), label_valid.to(device)
        data_train, label_train = dataset.data, dataset.label
        data_train, label_train = data_train.to(device), label_train.to(device)
        feature_dim = dataset.feature_dim
        trainloader = DataLoader(dataset, batch_size=args['batch_loss'], shuffle=True)
        weighted_avg = torch.zeros(1, feature_dim + 1).double().to(device)

        # initialize CBO optimizer
        if args['num_processes'] == 1 or device == 'cuda':
            optimizer = CBO_optimizer(
                num=args['num_particle'],
                dim=feature_dim+1,
                drift=args['drift'],
                noise=args['noise'],
                temp=args['temperature'],
                timestep=args['timestep'],
                tol=args['tolerance'],
                seed=args['seed'],
                batch_avg=args['batch_avg'],
                avg_choice=args['avg_choice'],
                noise_choice=args['noise_choice'],
                device=device,
                lam_reg=args['reg'],
                batch_loss=args['batch_loss'])
        else:
            optimizer = MultiprocessCBO(
                num_process=args['num_processes'],
                rank=rank,
                num=args['num_particle'],
                dim=feature_dim+1,
                drift=args['drift'],
                noise=args['noise'],
                temp=args['temperature'],
                timestep=args['timestep'],
                tol=args['tolerance'],
                seed=args['seed'],
                batch_avg=args['batch_avg'],
                avg_choice=args['avg_choice'],
                noise_choice=args['noise_choice'],
                device=device,
                lam_reg=args['reg'],
                batch_loss=args['batch_loss'])
        
        # optimizing...
        if rank == 0:
            print("Training...")
        epo_flag = False
        time_start = time.time()
        for i in range(args['epoch']):
            for train_batch, batch in enumerate(trainloader):
                if rank == 0:
                    t1 = time.time()
                data, label = batch_loader(batch)
                data, label = data.to(device), label.to(device)
                former = copy.deepcopy(weighted_avg)
                weighted_avg, noise_flag = optimizer.forward(data, label, args['problem'], weighted_avg)

                # synchronize(args['num_processes'])
                if rank == 0:
                    t2 = time.time()
                
                # noise_flag: whether to introduce BM in CBO
                # log recent results
                '''
                if noise_flag and rank == 0:
                    write_res('./results/train_loss_NCBO_logistic_log_loss_' + str(args['num_particle']) + '.npy', log_loss_train)
                    write_res('./results/train_loss_NCBO_logistic_log_acc_' + str(args['num_particle']) + '.npy', log_acc_train)
                    write_res('./results/loss_NCBO_logistic_log_loss_' + str(args['num_particle']) + '.npy', log_loss)
                    write_res('./results/loss_NCBO_ori_logistic_log_acc_' + str(args['num_particle']) + '.npy', log_acc)
                '''

                # validation
                u = weighted_avg[:-1]
                v = weighted_avg[-1:]
                pred_train = batch_logistic_class(data_train, u, v)
                pred_train_cnt = (pred_train == label_train).sum().double()
                loss_train = test_logistic_l0(data_train, label_train, u, v, args['reg'])
                log_loss_train.append(loss_train.item())
                log_acc_train.append(pred_train_cnt.item() / data_train.size(0))
                pred = batch_logistic_class(data_valid, u, v)
                pred_cnt = (pred == label_valid).sum().double()
                loss = test_logistic_l0(data_valid, label_valid, u, v, args['reg'])
                log_loss.append(loss.item())
                log_acc.append(pred_cnt.item() / data_valid.size(0))
                if rank == 0:
                    print("epoch: {:d} | iter: {:d} | loss: {:f} | validation precision: {:f}".format(i, train_batch, loss.item(), pred_cnt.item() / data_valid.size(0)))
                
                synchronize(args['num_processes'])

                if rank == 0:
                    t3 = time.time()
                    print("training time: {:f}".format(t2 - t1))
                    print("test time: {:f}".format(t3 - t2))

                # check whether to reach the stopping criterion
                delta_error = torch.norm(weighted_avg - former)
                err = delta_error * delta_error / weighted_avg.size(0)
                if err <= args['tolerance_stop']:
                    # print(former_avg)
                    # average, variance = var(former_avg)
                    # print(np.linalg.norm(former_avg), average, variance)
                    if rank == 0:
                        print("Consensus!")
                    epo_flag = True
                    break
            if epo_flag:
                optimizer._initialize_particles()
                break
        time_end = time.time()
        if rank == 0:
            print("total time: {:f}".format(time_end - time_start))
            print(min(log_loss_train))
            print(max(log_acc_train))
            print(min(log_loss))
            print(max(log_acc))
            write_res('./results/train_loss_RCBO_logistic_log_loss_' + str(args['num_particle']) + '.npy', log_loss_train)
            write_res('./results/train_loss_RCBO_logistic_log_acc_' + str(args['num_particle']) + '.npy', log_acc_train)
            write_res('./results/loss_RCBO_logistic_log_loss_' + str(args['num_particle']) + '.npy', log_loss)
            write_res('./results/loss_RCBO_logistic_log_acc_' + str(args['num_particle']) + '.npy', log_acc)
            print("Results written!")
        sys.exit("break!")                  
    elif args['dataset'] == "picture_reconstruction":
        data, label, theta = dataset.data, dataset.label, dataset.theta
        weighted_avg = torch.zeros(theta.size(0)).double()

        # initialize CBO optimizer
        if args['num_processes'] == 1:
            optimizer = CBO_optimizer(
                num=args['num_particle'],
                dim=theta.size(0),
                drift=args['drift'],
                noise=args['noise'],
                temp=args['temperature'],
                timestep=args['timestep'],
                tol=args['tolerance'],
                seed=args['seed'],
                batch_avg=args['batch_avg'],
                avg_choice=args['avg_choice'],
                noise_choice=args['noise_choice'],
                device=device,
                lam_reg=args['reg'],
                batch_loss=args['batch_loss'])
        else:
            optimizer = MultiprocessCBO(
                num_process=args['num_processes'],
                rank=rank,
                num=args['num_particle'],
                dim=theta.size(0),
                drift=args['drift'],
                noise=args['noise'],
                temp=args['temperature'],
                timestep=args['timestep'],
                tol=args['tolerance'],
                seed=args['seed'],
                batch_avg=args['batch_avg'],
                avg_choice=args['avg_choice'],
                noise_choice=args['noise_choice'],
                device=device,
                lam_reg=args['reg'],
                batch_loss=args['batch_loss'])
        optimizer._normalize_particles()
        optimizer.X *= 179.0
        
        # optimizing...
        if rank == 0:
            print("Training...")
        for i in range(args['epoch']):
            if rank == 0:
                t1 = time.time()
            former = copy.deepcopy(weighted_avg)
            weighted_avg, noise_flag = optimizer.forward(data, label, args['problem'], weighted_avg)

            synchronize(args['num_processes'])
            if rank == 0:
                t2 = time.time()

            # validation
            acc = theta_acc(weighted_avg, theta)
            loss = loss_recons(weighted_avg, data, label)
            log_loss.append(loss)
            log_acc.append(acc)
            if rank == 0:
                print("epoch: {:d} | loss: {:f} | validation error: {:f}".format(i, loss, acc))
            
            synchronize(args['num_processes'])

            if rank == 0:
                t3 = time.time()
                print("training time: {:f}".format(t2 - t1))
                print("test time: {:f}".format(t3 - t2))

            # check whether to reach the stopping criterion
            delta_error = torch.norm(weighted_avg - former)
            err = delta_error ** 2 / weighted_avg.size(0)
            print("error: ", err)
            if err <= args['tolerance_stop']:
                optimizer._initialize_particles()
                # print(former_avg)
                # average, variance = var(former_avg)
                # print(np.linalg.norm(former_avg), average, variance)
                if rank == 0:
                    print("Consensus!")
                break
        if rank == 0:
            write_res('./results/pic_loss_' + str(args['num_particle']) + '.npy', log_loss)
            write_res('./results/pic_acc_' + str(args['num_particle']) + '.npy', log_acc)
            print("Results written!")
        sys.exit("break!")
    elif args['dataset'] == "compressive_sensing":
        # data for compressive sensing
        data, label, observe, initial, L = dataset.data, dataset.label, dataset.observe, dataset.initial, dataset.L
        n_dim = dataset.sig_dim

        # initialize CBO optimizer
        if rank == 0:
            print("initialize CBO optimizer...")
        if args['num_processes'] == 1:
            optimizer = CBO_optimizer(
                num=args['num_particle'],
                dim=n_dim,
                drift=args['drift'],
                noise=args['noise'],
                temp=args['temperature'],
                timestep=args['timestep'],
                tol=args['tolerance'],
                seed=args['seed'],
                batch_avg=args['batch_avg'],
                avg_choice=args['avg_choice'],
                noise_choice=args['noise_choice'],
                device=device,
                lam_reg=args['reg'],
                batch_loss=args['batch_loss'])
        else:
            optimizer = MultiprocessCBO(
                num_process=args['num_processes'],
                rank=rank,
                num=args['num_particle'],
                dim=n_dim,
                drift=args['drift'],
                noise=args['noise'],
                temp=args['temperature'],
                timestep=args['timestep'],
                tol=args['tolerance'],
                seed=args['seed'],
                batch_avg=args['batch_avg'],
                avg_choice=args['avg_choice'],
                noise_choice=args['noise_choice'],
                device=device,
                lam_reg=args['reg'],
                batch_loss=args['batch_loss'])

        # start training
        if rank == 0:
            print("start training...")
        for j in range(args['repeat_time']):
            A = data[j]
            sig = label[j]
            b = observe[j]

            # for initial value
            optimizer._initialize_particles(j)
            X_0 = initial[j]
            gamma = math.sqrt(2.0 * args['reg'] / (L[j] + args['timestep']))
            print("L: ", L[j])
            print("gamma: ", gamma)
            loss = loss_cs(X_0, A, b, args['reg'])
            acc = acc_cs(X_0, sig)
            if rank == 0:
                print("FISTA done!")
                print("FISTA loss: ", loss)
                print("FISTA accuracy: ", acc)
            # sys.exit("sys exit: FISTA done!")

            # customize initialization for CBO
            # optimizer._custom_initialization(X_0, args['cs_std'])
            # weighted_avg = X_0.squeeze(1)
            weighted_avg = torch.zeros(X_0.size(0)).double()
            # synchronize(args['num_processes'])

            # optimizing...
            if rank == 0:
                print("training... | idx: {:d}".format(j))
            for epoch in range(args['epoch']):
                # CBO descent
                former = copy.deepcopy(weighted_avg)
                weighted_avg, noise_flag = optimizer.forward(A, b, args['problem'], weighted_avg)
                
                # try projection operation after several CBO steps to preserve the structure
                if epoch > 0 and epoch % 5 == 0:
                    # projection
                    proj_index = (weighted_avg >= gamma).double()
                    weighted_avg = weighted_avg * proj_index
                    # optimizer.X = optimizer.X * proj_index
                    print("projection!", (weighted_avg < gamma).sum().item())
                
                # log recent results
                if noise_flag and rank == 0:
                    write_res('./results/cs_log_loss_' + str(args['num_particle']) + '.npy', log_loss)
                    write_res('./results/cs_log_acc_' + str(args['num_particle']) + '.npy', log_acc)

                # validation
                loss = loss_cs(weighted_avg.unsqueeze(1), A, b, args['reg'])
                acc = acc_cs(weighted_avg.unsqueeze(1), sig)
                log_loss.append(loss)
                log_acc.append(acc)
                if rank == 0:
                    print("loss: ", loss)
                    print("idx: {:d} | epoch: {:d} | error: {:f}".format(j, epoch, acc))

                # check whether to reach the stopping criterion
                delta_error = torch.norm(weighted_avg - former)
                err = delta_error ** 2 / weighted_avg.size(0)
                # print("delta error: ", delta_error)
                if err <= args['tolerance_stop']:
                    if rank == 0:
                        print("Consensus!")
                    break
                
                synchronize(args['num_processes'])

            if rank == 0:
                print("idx / n_data : {:d} / {:d} is done".format(j, args['repeat_time']))
            optimizer._initialize_particles(j + 1)
            synchronize(args['num_processes'])
        sys.exit("sys exit: break!")
    elif args['dataset'] == "mnist":
        # preparing data...
        trainloader = DataLoader(data_train, batch_size=args['batch_loss'], shuffle=True)
        testloader = DataLoader(data_test, batch_size=args['batch_loss'], shuffle=False)
        if args['problem'] == 'one_module':
            feature_dim = 784*10 + 10
        elif args['problem'] == 'two_module':
            feature_dim = 784*50 + 50 + 50*10 + 10
        weighted_avg = torch.zeros(1, feature_dim).double().to(device)

        # initialize CBO optimizer
        if args['num_processes'] == 1 or device == 'cuda':
            optimizer = CBO_optimizer(
                num=args['num_particle'],
                dim=feature_dim,
                drift=args['drift'],
                noise=args['noise'],
                temp=args['temperature'],
                timestep=args['timestep'],
                tol=args['tolerance'],
                seed=args['seed'],
                batch_avg=args['batch_avg'],
                avg_choice=args['avg_choice'],
                noise_choice=args['noise_choice'],
                device=device,
                lam_reg=args['reg'],
                batch_loss=args['batch_loss'])
        else:
            optimizer = MultiprocessCBO(
                num_process=args['num_processes'],
                rank=rank,
                num=args['num_particle'],
                dim=feature_dim,
                drift=args['drift'],
                noise=args['noise'],
                temp=args['temperature'],
                timestep=args['timestep'],
                tol=args['tolerance'],
                seed=args['seed'],
                batch_avg=args['batch_avg'],
                avg_choice=args['avg_choice'],
                noise_choice=args['noise_choice'],
                device=device,
                lam_reg=args['reg'],
                batch_loss=args['batch_loss'])
        
        # same initialzation as weight initialization in Pytorch
        if args['problem'] == 'two_module':
            optimizer._kaiming_uniform_initialization()
        elif args['problem'] == 'one_module':
            optimizer._kaiming_1_uniform_initialization()

        # optimizing...
        if rank == 0:
            print("Training...")
        epo_flag = False
        time_start = time.time()
        for i in range(args['epoch']):
            for train_batch, (data, label) in enumerate(trainloader):
                if rank == 0:
                    t1 = time.time()
                former = copy.deepcopy(weighted_avg)
                data, label = data.to(device), label.to(device)
                weighted_avg, noise_flag = optimizer.forward(data, label, args['problem'], weighted_avg)

                # log recent results
                if (noise_flag and rank == 0) or (i % 5 == 0):
                    write_res('./results_mnist/log_loss_train_' + str(args['num_particle']) + '.npy', log_loss_train)
                    # write_res('./results_mnist/log_loss_ave_train_' + str(args['num_particle']) + '.npy', log_loss_ave_train)
                    write_res('./results_mnist/log_acc_train' + str(args['num_particle']) + '.npy', log_acc_train)
                    write_res('./results_mnist/nnew_noise_TM_log_loss_' + str(args['num_particle']) + '.npy', log_loss)
                    write_res('./results_mnist/nnew_noise_TM_log_acc_' + str(args['num_particle']) + '.npy', log_acc)
                
                # synchronize(args['num_processes'])
                # print train loss & acc
                if args['problem'] == 'one_module':
                    loss_train, acc_train = OneModule_test(data, label, weighted_avg, device)
                elif args['problem'] == 'two_module':
                    loss_train, acc_train = TwoModule_test(data, label, weighted_avg, device)
                log_loss_train.append(loss_train)
                # log_loss_ave_train.append(train_loss_ave)
                log_acc_train.append(acc_train)
                if rank == 0:
                    print("epoch: {:d} | iter: {:d} | train loss: {:f} | train precision: {:f}".format(i, train_batch, loss_train.item(), acc_train.item()))
                    t2 = time.time()
                    print("training time: {:f}".format(t2 - t1))

                # validation
                loss_sum = 0
                acc_sum = 0
                loss_ave = 0
                t3 = time.time()
                for val_batch, (data_val, label_val) in enumerate(testloader):
                    data_val, label_val = data_val.to(device), label_val.to(device)
                    if args['problem'] == 'one_module':
                        loss_val, acc_val = OneModule_test(data_val, label_val, weighted_avg, device)
                    elif args['problem'] == 'two_module':
                        loss_val, acc_val = TwoModule_test(data_val, label_val, weighted_avg, device)
                    loss_sum += loss_val.item() * label_val.size(0)
                    acc_sum += acc_val.item() * label_val.size(0)
                loss_sum /= len(testloader.dataset)
                acc_sum /= len(testloader.dataset)
                log_loss.append(loss_sum)
                log_acc.append(acc_sum)
                t4 = time.time()
                if rank == 0:
                    # print("loss: ", loss)
                    print("epoch: {:d} | loss: {:f} | validation precision: {:f}".format(i, loss_sum, acc_sum))
                    print("test time: {:f}".format(t3 - t2))

                synchronize(args['num_processes'])
                
                # check whether to reach the stopping criterion
                delta_error = torch.norm(weighted_avg - former)
                err = delta_error**2 / weighted_avg.size(0)
                if err <= args['tolerance_stop']:
                    res_loss[0] = loss_sum
                    res_acc[0] = acc_sum
                    # print(former_avg)
                    # average, variance = var(former_avg)
                    # print(np.linalg.norm(former_avg), average, variance)
                    if rank == 0:
                        print("Consensus!")
                    epo_flag = True
                    break
                
            synchronize(args['num_processes'])

        if rank == 0:
            time_end = time.time()
            print('epoch time', time_end - time_start)
            # sys.exit("time!")
            write_res('./results_mnist/log_loss_train_' + str(args['num_particle']) + '.npy', log_loss_train)
            write_res('./results_mnist/log_acc_train' + str(args['num_particle']) + '.npy', log_acc_train)
            write_res('./results_mnist/nnew_noise_TM_log_loss_' + str(args['num_particle']) + '.npy', log_loss)
            write_res('./results_mnist/nnew_noise_TM_log_acc_' + str(args['num_particle']) + '.npy', log_acc)
            print("Results written!")
        sys.exit("break!")
    elif args['dataset'] == 'ras':
        # initilize...
        B = torch.tensor(args['B']).double()
        C = torch.tensor(args['C']).double()
        weighted_avg = torch.zeros(1, args['ras_dim']).double()

        # initialize CBO optimizer
        if args['num_processes'] == 1:
            optimizer = CBO_optimizer(
                num=args['num_particle'],
                dim=args['ras_dim'],
                drift=args['drift'],
                noise=args['noise'],
                temp=args['temperature'],
                timestep=args['timestep'],
                tol=args['tolerance'],
                seed=args['seed'],
                batch_avg=args['batch_avg'],
                avg_choice=args['avg_choice'],
                noise_choice=args['noise_choice'],
                device=device,
                lam_reg=args['reg'],
                batch_loss=args['batch_loss'])
        else:
            optimizer = MultiprocessCBO(
                num_process=args['num_processes'],
                rank=rank,
                num=args['num_particle'],
                dim=args['ras_dim'],
                drift=args['drift'],
                noise=args['noise'],
                temp=args['temperature'],
                timestep=args['timestep'],
                tol=args['tolerance'],
                seed=args['seed'],
                batch_avg=args['batch_avg'],
                avg_choice=args['avg_choice'],
                noise_choice=args['noise_choice'],
                device=device,
                lam_reg=args['reg'],
                batch_loss=args['batch_loss'])
        
        # optimizing...
        if rank == 0:
            print("Training...")
        log_gap_list = list()
        min_gap_list = list()
        success_sum = 0
        for times in range(args['repeat_time']):
            log_gap = list()
            set_random_seed(times)
            optimizer._uniform_initialization_particles(times)
            acc_flag = False
            min_gap = 1e5
            for i in range(args['epoch']):
                if rank == 0:
                    t1 = time.time()
                former = copy.deepcopy(weighted_avg)
                weighted_avg, noise_flag = optimizer.forward(B, C, args['problem'], weighted_avg)
                flag_acc = 0

                if rank == 0:
                    t2 = time.time()

                # validation
                pred = rastrigin(weighted_avg, B, C)
                gap = torch.log(abs(pred - C))
                log_gap.append(gap)
                min_gap = min(min_gap, torch.norm(weighted_avg - B)**2)
                if (abs(weighted_avg - B) < 0.25).all():
                    flag_acc = 1
                    acc_flag = True
                if rank == 0:
                    # print("loss: ", loss)
                    print("time: {:d} | epoch: {:d} | loss: {:f} | success: {:d}".format(times, i, gap.item(), flag_acc))
                
                synchronize(args['num_processes'])

                if rank == 0:
                    t3 = time.time()
                    print("training time: {:f}".format(t2 - t1))
                    print("test time: {:f}".format(t3 - t2))
                # if pred <= 0 or flag_acc:
                #     break
                #     print(weighted_avg)

            log_gap_list.append(log_gap)
            min_gap_list.append(min_gap)
            success_sum += flag_acc    
            if rank == 0:
                if acc_flag:
                    print("success!")
                else:
                    print("failed!")
                write_res('./results/NNCBO_rastrigin_log_loss_' + str(args['num_particle']) + '_' + str(args['ras_dim']) + '.npy', log_gap_list)
                write_res('./results/NNCBO_rastrigin_min_gap_' + str(args['num_particle']) + '_' + str(args['ras_dim']) + '.npy', min_gap_list)
                print("Results written!")
        print(success_sum)
        sys.exit("break!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Numerical Test for Consensus-based Global Optimization Method',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # problem setting
    parser.add_argument('-d', '--dataset', type=str, default=None, help='the dataset we use')
    parser.add_argument('-p', '--problem', type=str, default=None, help='the problem we want to optimize')
    parser.add_argument('-g', '--reg', type=float, default=None, help='the parameter for the regularization term')

    # for compressive sensing problems
    parser.add_argument('-nd', '--n-dim', type=int, default=None, help='the dimension of the original signal')
    parser.add_argument('-sp', '--sparsity', type=int, default=None, help='the sparsity of the original signal (1/2)')
    parser.add_argument('-nc', '--cs-std', type=float, default=None, help='the std of the noise for customized initialization in cs problem')
    parser.add_argument('-tt', '--repeat-time', type=int, default=1, help='the number of experiments of different (n, s)')
    parser.add_argument('-gf', '--reg-fista', type=float, default=0.1, help='the parameter for the regularization term in FISTA')
    parser.add_argument('-it', '--iteration', type=int, default=100, help='the number of iterations for FISTA')
    parser.add_argument('-tf', '--tolerance-fista', type=float, default=0.01, help='the tolerance for the stopping in FISTA')
    # parser.add_argument('-ic', '--iteration-CBO', type=int, default=1, help='the iteration for CBO before the projection')
    parser.add_argument('-mu', '--mu-proj', type=float, default=0.0, help='the parameter for the projection operation')

    # for rastrigin problem
    parser.add_argument('-dd', '--ras-dim', type=int, default=10, help='the dim for the rastrigin function')
    parser.add_argument('-b', '--B', type=float, default=0.0, help='the argmin of the rastrigin function')
    parser.add_argument('-c', '--C', type=float, default=0.0, help='the min of the rastrigin function')

    # parameters for CBO
    parser.add_argument('-np', '--num-particle', type=int, default=None, help='the number of particles we use')
    parser.add_argument('-r', '--drift', type=float, default=None, help='the parameter of drift for consensus-based model')
    parser.add_argument('-n', '--noise', type=float, default=None, help='the parameter of noise for consensus-based model')
    parser.add_argument('-t', '--temperature', type=float, default=None, help='the temperature for the weighted average')
    parser.add_argument('-lr', '--timestep', type=float, default=None, help='the timestep for consensus-based model')
    parser.add_argument('-ba', '--batch-avg', type=int, default=None, help='the batch size we use to calculate the weighted average')
    parser.add_argument('-bl', '--batch-loss', type=int, default=None, help='the batch size we use to calculate the loss function')
    parser.add_argument('-e', '--epoch', type=int, default=50, help='the iteration number for consensus-based optimization problem')
    parser.add_argument('-tol', '--tolerance', type=float, default=1e-3, help='the tolerance for the difference of avg to add noises')
    parser.add_argument('-ts', '--tolerance-stop', type=float, default=1e-8, help='the tolerance for the stopping criteria')
    parser.add_argument('-av', '--avg-choice', action="store_false", help='whether to choose entropy rule to update avg, otherwise argmin')
    parser.add_argument('-nn', '--noise-choice', action="store_true", help='whether to add noise term in CBO')

    # general setting
    parser.add_argument('-s', '--seed', type=int, default=0, help='the random seed')
    parser.add_argument('-gp', '--gpu', action="store_true", help='whether to apply gpu training')
    parser.add_argument('-npp', '--num-processes', type=int, default=32, help='the number of processes for multiprocessing optimization')
    parser.add_argument('-mi', '--master-ip', type=str, default='127.0.0.1')
    parser.add_argument('-mp', '--master-port', type=str, default='12345')
    
    args = parser.parse_args()
    args = setup(args)

    if args['num_processes'] == 1:
        main(0, args)
    else:
        mp = torch.multiprocessing.get_context('spawn')
        procs = []
        for rank in range(args['num_processes']):
            procs.append(mp.Process(target=launch_a_process, args=(rank, args, main), daemon=True))
            procs[-1].start()
        for p in procs:
            p.join()


#################################### not used #################################
'''
loss_min = loss_map[args.problem](data_valid, label_valid, X[0, :-1], X[0, -1:], args.reg, args.batch_loss)
min_idx = 0
for j in range(X.shape[0]):
    loss = loss_map[args.problem](data_valid, label_valid, X[j, :-1], X[j, -1:], args.reg, args.batch_loss)
    if loss <= loss_min:
        loss_min = loss
        min_idx = j
'''
'''
m = 3000
n = 8000
s = math.floor(n / 100)
A, sig, b = generate_cs(m, n, s, args.seed)
noi = math.sqrt(args.noise)
former_avg = np.zeros((sig.shape[0], 1))
X = []
X_0 = A.T.dot(b)
for i in range(args.num_particle):
    # np.random.seed(args.seed)
    y = np.random.normal(loc=0.0, scale=noi,size=(sig.shape[0], 1))
    x = X_0 + y
    X.append(x.reshape(sig.shape[0], ))
X = np.array(X).astype(float)
np.random.seed(args.seed)
noise = np.random.normal(loc=0.0, scale=noi, size=(args.num_particle, X.shape[1]))
for epoch in range(args.epoch):
    former = copy.deepcopy(former_avg)
    X, former_avg = consensus(X, A, b, args.problem, args.batch_avg, args.drift, noi, args.temperature, args.reg, args.timestep, args.tolerance, former_avg, noise)
    acc = acc_cs(former_avg, sig)
    print("iter: {:d} | acc: {:f}".format(epoch, acc))
    loss = np.linalg.norm(former_avg - former) / max(1, np.linalg.norm(former_avg))
    if loss <= args.tolerance_cs:
        break
'''
'''
X = []
for par in range(args.num_particle):
    # np.random.seed(args.seed)
    y = np.random.normal(loc=0.0, scale=1.0, size=(sig.shape[0], 1))
    x = X_0 + y
    X.append(x.reshape(sig.shape[0], ))
X = np.array(X).astype(float)
'''
'''
# FISTA for starting pt (from l1 regularized problem)
X_0 = torch.t(A).mm(b)
# L = torch.norm(A) ** 2
L,  = np.linalg.eig((torch.t(A).mm(A)).numpy())
L = torch.from_numpy(L).double()
L = torch.max(L)
gamma = torch.sqrt(2 * args['reg'] / (L + args['mu_proj']))
X_0 = FISTA(X_0, A, b, L, args['reg_fista'], args['iteration'], args['tolerance_fista'])
'''
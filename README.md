# Consensus-based-Optimization
An Implementation for CBO algorithm

# Environment
`conda create -n CBO-env python=3.6`<br>
`source activate CBO-env`<br>
`conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`<br>

# How to
## For CBO optimizer
If you are only looking for the implementation of the CBO optimizer, please refer to the `CBO.py`. The original version and multiprocessing version are both implemented, and they all provide the customization for the noise term and 'restart' operation. 
### Initialization
To initialize the original CBO optimizer, you will be asked to provide the following arguments:
```
-num, number of particles for CBO
-dim, dimension of the parameter for the optimization problem
-drift, parameter for the drift term in CBO
-noise, square of the parameter for the noise term in CBO
-temp, parameter for the inverse of temperature in CBO
-timestep, timestep for the discretized CBO model
-tol, tolerance to make all the particles to do an indepent BM with variance noise when the weighted average stops updating
-seed, random seed
-batch_avg, the random batch size for calculating the weighted average
-avg_choice, how to update the weighted average (by Gibbs distribution or argmin of loss)
-noise_choice, whether to apply the noise term in the original CBO scheme
-lam_reg, parameter for the regularization term in the optimization if any (default: None)
-batch_loss, random batch size for calculating the loss function if any (default: None)
-gpu, whether to apply gpu training (default: False)
```
To initialize the multiprocessing CBO, you will be asked to provide addtional `num_process` for the number of processes, `rank` for the process (0 - `num_process-1`). The `-num` only stands for the number of particles of a single process, the total number would be `num_process * num`. Others arguments are all the same as the original CBO.
### Particle Operation
Initial particles are generated from standard normal distribution. Uniform generation, simple customization and normalization are all implemented.
### Optimization Process
For updating from the k-th weighted average to the (k+1)-th one, you can simply use
```
weighted_avg_new, noise_flag = optimizer.forward(data, label, pro, weighted_avg_old)
```
Here `data`, `label` and `pro` are parameters for the loss function, which you can customize. The `noise_flag` returns the bool value of whether apply the additional BM.<br>
Different loss functions are added to the `batch_loss_map`，where `pro` would be the key, and valus are loss functions. All loss functions are taking
```
-X_update, -data, -label, -temp, -lem_reg, -avg_choice
```
where `X_update` is the random batch for calculating the weighted average. All loss functions are returning
```
-num, the sum of X(i-th particle in the random batch) * exp(-temp*L(X_i))
-dom, the sum of exp(-temp*L(X_i))
```

You can add your own loss funtion to the `batch_loss_map`, or modify the code in `def avg`.<br>
You need to implement the stopping criterion by comparing two most recent weighted averages in your own `main.py`.
## For Numerical Experiments
### Prepare Datasets
You can refer to `load_data.py` to see details about data prepareation for different numerical experiments.<br>
For the Gisette dataset, you first need to unzip the `gisette` zip.
### Objective Functions
You can refer to corresponding `python files` to see details about objective functions for different numerical experiments.<br>
Moreover, in `Neural_Network.py`, you can also test the performance of other optimization algorithms such as SGD, Adam on neural network optimization on the MNIST dataset.
### Optimization Process
You can refer to `main.py` to see details about optimization process for different numerical experiments.

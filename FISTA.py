import math
import numpy as np
import torch

def FISTA(x, A, b, L, lam, num_iter, tol):
    x_pre = x
    x_next = x
    t_pre = 1
    M = A.T.dot(A)
    N = A.T.dot(b)
    for iter in range(num_iter):
        t_next = (1 + math.sqrt(1 + 4 * t_pre * t_pre)) / 2
        y = x_pre + t_pre * (x_next - x_pre) / t_next
        x_pre = x_next
        t_pre = t_next
        dev = M.dot(y) - N
        temp = y - dev / L
        pos_op = (abs(temp) - lam / L)
        pos_op = np.maximum(pos_op, 0)
        x_next = pos_op * (np.sign(temp))
        # x_next = min(1e10, max(-1e10, x_next))
        flag = np.linalg.norm(x_next - x_pre) / max(1, np.linalg.norm(x_next))
        print(iter, flag)
        if flag < tol:
            break
    print("FISTA iteration: ", iter)
    return x_next


if __name__ == "__main__":
    A = torch.randn(10, 10)
    b = torch.ones(10, 1)
    X_0 = torch.t(A).mm(b)
    L = torch.norm(A) ** 2
    X_0 = FISTA(X_0, A, b, L, 0.05, 1000, 10)
    print(X_0.size())
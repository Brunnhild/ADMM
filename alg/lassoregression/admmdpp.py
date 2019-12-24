import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from alg.utils import *
#from scipy.interpolate import spline


def ADMM(A,
         b,
         alpha=0.1,
         beta=0.01,
         show_x=True,
         show_graph=True,
         log_int=1,
         show_penalty=True,
         max_step=-1):
    n = A.shape[1]
    x = np.random.rand(n, 1)
    z = np.random.rand(n, 1)
    landa = np.random.rand(n, 1)
    k = 0
    if show_x:
        print("x starts with: ")
        print(x)
    red = np.linalg.norm(x, ord=1) + (alpha / 2) * np.linalg.norm(A @ x - b)**2
    print('The initial target value is %f' % (red))
    obj = []
    I = np.identity(n)
    one = np.ones((n, 1))
    zero = np.zeros((n, 1))
    zeros = np.zeros((n, n))
    D = np.eye(n)
    D *= beta
    while (1):
        temp1 = np.linalg.inv(alpha * (A.T @ A) + D)
        temp2 = alpha * (A.T @ b) + (D @ z) - landa
        xx = temp1 @ temp2
        
        zz = np.sign(xx + np.linalg.inv(D) @ landa) * np.maximum(
            np.abs(xx + np.linalg.inv(D) @ landa) - np.linalg.inv(D) @ one,
            zero)
        landalanda = landa + D @ (xx - zz)

        red = np.linalg.norm(
            xx, ord=1) + (alpha / 2) * np.linalg.norm(A @ xx - b)**2
        if k % log_int == 0:
            obj.append(red)
            print('The %dth iteration, target value is %f' % (k, red))

        if k == max_step:
            break

        if stop(I, xx, -I, zz, z, zero, landalanda, alpha):
            break
        else:
            r = xx - zz
            s = -1 * alpha * (zz - z)
            u = 10
            t_incr = 2
            t_decr = t_incr
            tmp = np.abs(r) - u * np.abs(s)
            tmp = np.where(tmp > 0, t_incr, 1)
            D *= tmp
            tmp = np.abs(s) - u * np.abs(r)
            tmp = np.where(tmp > 0, 1 / t_decr, 1)
            D *= tmp
            x = np.copy(xx)
            z = np.copy(zz)
            landa = np.copy(landalanda)
            if show_x:
                print('The current x is: ', x)
            k += 1

    print("Final x: ", xx)
    print("Total steps: ", k)

    if show_graph:
        draw(obj, log_int)

    return xx, k, obj


def main():
    x = np.array([[1.]])
    z = np.array([[1.]])
    landa = np.array([[1.]])
    A = np.array([[1.]])
    b = np.array([[.2]])
    alpha = 0.1
    beta = 0.01
    D = np.array([[0.5]])
    xx, count, obj = ADMM(x, alpha, landa, A, b, beta, z, D)

    draw(obj)
    print("最终迭代结果: x：", xx)
    print("共进行了", count, "次迭代")

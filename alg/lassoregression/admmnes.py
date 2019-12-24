#LASSO Regression

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from alg.utils import *
#from scipy.interpolate import spline


def ADMM(A, b, alpha=0.1, beta=0.01, show_x=True, show_graph=True, log_int=1):
    n = A.shape[1]
    nl = 0
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
    zero = np.zeros((n))
    zero = zero.T
    yz = np.copy(z)
    yx = np.copy(x)

    while (1):
        nl_new = (1 + sqrt(1 + 4 * nl * nl)) / 2
        gamma = (1 - nl) / nl_new
        temp1 = np.linalg.inv(alpha * (A.T @ A) + beta * I)
        temp2 = alpha * (A.T @ b) + beta * z - landa
        y_xx = temp1 @ temp2
        xx = (1 - gamma) * y_xx + gamma * yx
        y_zz = np.sign(xx + landa / beta) * np.maximum(
            np.abs(xx + landa / beta) - 1 / beta, 0)
        zz = (1 - gamma) * y_zz + gamma * yz
        landalanda = landa + beta * (xx - zz)
        red = np.linalg.norm(
            xx, ord=1) + (alpha / 2) * np.linalg.norm(A @ xx - b)**2
        if k % log_int == 0:
            obj.append(red)
            print('The %dth iteration, target value is %f' % (k, red))
        #A,xx,B,zz,z,b,landalanda,alpha
        if stop(I, xx, -I, zz, z, zero, landalanda, alpha):
            break
        else:
            x = xx
            z = zz
            yx = y_xx
            yz = y_zz
            landa = landalanda
            nl = nl_new
            if show_x:
                print('The current x is: ', x)
        k = k + 1

    print("Final x: ", xx)
    print("Total steps: ", k)

    if show_graph:
        draw(obj, log_int)

    return xx, k, obj


def main():

    yx = np.array([[1.]])
    yz = np.array([[1.]])
    landa = np.array([[1.]])
    A = np.array([[1.]])
    b = np.array([[.2]])
    alpha = 0.1
    error = 0.01
    beta = 0.01
    #alpha,landa,A,b,beta,yx,yz
    xx, count, obj = ADMM(alpha, landa, A, b, beta, yx, yz)
    draw(obj)

    print("最终迭代结果: x：", xx)
    print("共进行了", count, "次迭代")


if __name__ == '__main__':
    main()

#LASSO Regression

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from alg.utils import stop
from alg.utils import draw
#from scipy.interpolate import spline


def PGM(x, alpha, landa, A, b, error):
    k = 0
    print("x的取值的迭代过程：")
    print(x)
    obj = []
    while (1):
        z = x - alpha * landa * np.dot(A.T, np.dot(A, x) - b)
        xx = np.copy(x)
        xx = np.sign(z) * np.maximum(np.linalg.norm(z) - alpha, 0)

        red = np.linalg.norm(
            xx, ord=1) + (alpha / 2) * np.linalg.norm(A @ xx - b)**2
        obj.append(red)
        print('The %dth iteration, target value is %f' % (k, red))

        e = np.linalg.norm(xx - x)
        if e < error:
            break
        else:
            x = np.copy(xx)
            print('The current x is: ', x)
        k += 1
    return xx, k, obj


def ADMM(A, b, alpha=0.1, beta=0.01, show_x=True, show_graph=True, log_int=1):
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
    zero = np.zeros(n)
    zero = zero.T

    while (1):
        #print(I)
        #此处的landa是向量
        temp1 = np.linalg.inv(alpha * (A.T @ A) + beta * I)
        #print(temp1)
        temp2 = alpha * (A.T @ b) + beta * z - landa

        xx = temp1 @ temp2
        #print(xx)
        zz = np.copy(z)

        zz = np.sign(xx + landa / beta) * np.maximum(
            np.abs(xx + landa / beta) - 1 / beta, 0)

        landalanda = landa + beta * (xx - zz)

        red = np.linalg.norm(
            xx, ord=1) + (alpha / 2) * np.linalg.norm(A @ xx - b)**2
        if k % log_int == 0:
            obj.append(red)
        if k % log_int == 0:
            print('The %dth iteration, target value is %f' % (k, red))
        '''
        xzk = np.vstack((x, z))
        xzkp = np.vstack((xx, zz))
        
        e = np.linalg.norm(xzk - xzkp)
        
        if e <= error:
            break
        '''
        if stop(I, xx, -I, zz, z, zero, landalanda, alpha):
            break
        else:
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
    #PGM:
    '''
    x = np.array([[1.]])
    landa = 0.5
    alpha = 0.1
    A = np.array([[1.]])
    b = np.array([[.2]])
    error = 0.001
    xx, count, obj = PGM(x, alpha, landa, A, b, error)
    print(obj)
    draw(obj)
    '''
    #ADMM

    x = np.array([[1.]])
    z = np.array([[1.]])
    landa = np.array([[1.]])
    A = np.array([[1.]])
    b = np.array([[.2]])
    alpha = 0.1
    error = 0.01
    beta = 0.01
    xx, count, obj = ADMM(x, alpha, landa, A, b, error, beta, z)
    draw(obj)

    print("最终迭代结果: x：", xx)
    print("共进行了", count, "次迭代")

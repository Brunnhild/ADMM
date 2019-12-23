#Ridge Regression

import numpy as np
from math import sqrt
import random
import matplotlib.pyplot as plt
#from scipy.interpolate import spline


def SD(x, alpha, landa, A, b, error):
    k = 0
    print("x的取值的迭代过程：")
    print(x)
    obj = []
    while (1):
        g = 2 * x + alpha * np.dot(A.T, np.dot(A, x) - b)
        xx = x - landa * g

        red = np.linalg.norm(xx)**2 + (alpha / 2) * np.linalg.norm(A @ xx -
                                                                   b)**2
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


def BFGS(x, xx, A, b, alpha, D, epsi, error):
    k = 0
    print("x的取值的迭代过程：")
    print(x)
    print(xx)
    obj = []
    while (1):
        p = xx - x

        delta_x = 2 * x + alpha * np.dot(A.T, np.dot(A, x) - b)
        delta_xx = 2 * xx + alpha * np.dot(A.T, np.dot(A, x) - b)
        q = delta_xx - delta_x

        tao = np.dot(np.dot(q.T, D), q)

        temp1 = p / np.dot(p.T, q)
        temp2 = np.dot(D, q) / tao
        v = temp1 - temp2

        part_2 = np.dot(p, p.T) / np.dot(p, q.T)
        part_3 = np.dot(np.dot(np.dot(D, q), q.T), D.T) / np.dot(
            np.dot(q.T, D), q)
        part_4 = epsi * np.dot(np.dot(tao, v), v.T)
        DD = D + part_2 - part_3 + part_4

        xxx = xx - alpha * np.dot(DD, delta_xx)

        red = np.linalg.norm(xxx)**2 + (alpha / 2) * np.linalg.norm(A @ xxx -
                                                                    b)**2
        obj.append(red)
        print('The %dth iteration, target value is %f' % (k, red))

        e = np.linalg.norm(xxx - xx)
        if e < error:
            break
        else:
            x = np.copy(xx)
            xx = np.copy(xxx)
            D = np.copy(DD)
            print('The current x is: ', xx)
        k += 1

    return xxx, k, obj


def ADMM(x, alpha, landa, A, b, error, beta, z):
    k = 0
    print("x的取值的迭代过程：")
    print(x)
    obj = []
    p = random.uniform(1.5, 1.8)
    I = np.identity(len(A[0]))
    while (1):
        part_1 = np.linalg.inv(alpha * (A.T @ A) + beta + I)
        part_2 = alpha * (A.T @ b) + beta * z - landa
        xx = part_1 @ part_2
        zz = (landa + beta * (xx * p - (1 - p) * z)) / (2 + beta)
        landalanda = landa + beta * (p * xx - (1 - p) * z - zz)

        xzk = np.vstack((x, z))
        xzkp = np.vstack((xx, zz))

        red = np.linalg.norm(
            xx, ord=1) + (alpha / 2) * np.linalg.norm(A @ xx - b)**2
        obj.append(red)
        print('The %dth iteration, target value is %f' % (k, red))

        e = np.linalg.norm(xzk - xzkp)
        if e <= error:
            break
        else:
            x = np.copy(xx)
            z = np.copy(zz)
            landa = np.copy(landalanda)
            print('The current x is: ', x)
        k += 1
    return xx, k, obj


def draw(obj):
    x = np.zeros(len(obj))
    for i in range(len(obj)):
        x[i] = i * 0.1
    '''
    x_new=np.linspace(x.min(),x.max(),300)
    obj_smooth=spline(x,obj,x_new)
    '''
    plt.scatter(x, obj, c='black')
    plt.plot(x, obj, linewidth=1)
    #plt.plot(x_new,obj_smooth,c='red')
    plt.ylabel('obj')
    plt.show()


def main():
    #SD:
    # x = np.array([[1.]])
    # alpha=0.1
    # landa=0.5
    # A = np.array([[1.]])
    # b = np.array([[.2]])
    # error=0.0001
    # xx,count,obj=SD(x,alpha,landa,A,b,error)
    #BFGS:

    # x = np.array([[1.]])
    # xx = np.array([[1.4]])
    # alpha=0.1
    # epsi=0.3
    # A = np.array([[1.]])
    # b = np.array([[.2]])
    # error=0.001
    # D = np.array([[1.5]])
    # xx,count,obj=BFGS(x,xx,A,b,alpha,D,epsi,error)

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


main()

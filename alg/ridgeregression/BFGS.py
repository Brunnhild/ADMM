#Ridge Regression

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from alg.utils import stop
from alg.utils import draw
#from scipy.interpolate import spline


def BFGS(A, b, alpha=0.1, epsi=0.3, error=0.001, show_x=True, show_graph=True, log_int=1):
    k = 0
    n = A.shape[1]
    x = np.random.rand(n, 1)
    xx = np.random.rand(n, 1)
    D = np.random.rand(n, n)
    if show_x:
        print("x starts with: ")
        print(x)
    obj = []
    while (1):
        p = xx - x

        delta_x = 2 * x + alpha * np.dot(A.T, np.dot(A, x) - b)
        delta_xx = 2 * xx + alpha * np.dot(A.T, np.dot(A, x) - b)
        q = delta_xx - delta_x

        tao = q.T @ D @ q

        temp1 = p / np.dot(p.T, q)
        temp2 = np.dot(D, q) / tao
        v = temp1 - temp2

        part_2 = np.dot(p, p.T) / np.dot(p, q.T)
        part_3 = np.dot(np.dot(np.dot(D, q), q.T), D.T) / np.dot(
            np.dot(q.T, D), q)
        part_4 = epsi * tao * v @ v.T
        DD = D + part_2 - part_3 + part_4

        xxx = xx - alpha * np.dot(DD, delta_xx)

        red = np.linalg.norm(xxx)**2 + (alpha / 2) * np.linalg.norm(A @ xxx -
                                                                    b)**2
        if k % log_int == 0:
            obj.append(red)
            print('The %dth iteration, target value is %f' % (k, red))

        e = np.linalg.norm(xxx - xx)
        if e < error:
            break
        else:
            x = np.copy(xx)
            xx = np.copy(xxx)
            D = np.copy(DD)
            if show_x:
                print('The current x is: ', x)
        k += 1

    print("Final x: ", xx)
    print("Total steps: ", k)
    
    if show_graph:
        draw(obj, log_int)
        
    return xx, k, obj

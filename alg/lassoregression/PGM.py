#LASSO Regression

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from alg.utils import stop
from alg.utils import draw
#from scipy.interpolate import spline


def PGM(A, b, alpha=0.1, error=0.001, show_x=True, show_graph=True, log_int=1):
    k = 0
    n = A.shape[1]
    x = np.random.rand(n, 1)
    landa = np.random.rand(n, 1)
    if show_x:
        print("x starts with: ")
        print(x)
    red = np.linalg.norm(x, ord=1) + (alpha / 2) * np.linalg.norm(A @ x - b)**2
    print('The initial target value is %f' % (red))
    obj = []
    while (1):
        z = x - alpha * landa * np.dot(A.T, np.dot(A, x) - b)
        xx = np.copy(x)
        xx = np.sign(z) * np.maximum(np.linalg.norm(z) - alpha, 0)

        red = np.linalg.norm(
            xx, ord=1) + (alpha / 2) * np.linalg.norm(A @ xx - b)**2
        if k % log_int == 0:
            obj.append(red)
        if k % log_int == 0:
            print('The %dth iteration, target value is %f' % (k, red))

        e = np.linalg.norm(xx - x)
        if e < error:
            break
        else:
            x = np.copy(xx)
            if show_x:
                print('The current x is: ', x)
        k += 1
    
    print("Final x: ", xx)
    print("Total steps: ", k)

    if show_graph:
        draw(obj, log_int)

    return xx, k, obj

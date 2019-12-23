import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def stop(A, xx, B, zz, z, b, landalanda, alpha):
    rr = A @ xx + B * zz - b
    ss = alpha * (A.T @ B @ (zz - z))
    n = xx.shape[0]
    epr = 0.001
    epa = 0.01
    epp = sqrt(n) * epa + epr * max(np.linalg.norm(A @ xx),
                                    np.linalg.norm(B @ zz), np.linalg.norm(b))
    epd = sqrt(n) * epa + epr * np.linalg.norm(A.T @ landalanda)
    if np.linalg.norm(rr, ord=2) <= epp and np.linalg.norm(ss, ord=2) <= epd:
        return True
    else:
        return False


def draw(obj, interval):
    x = np.zeros(len(obj))
    for i in range(len(obj)):
        x[i] = i * interval
    '''
    x_new=np.linspace(x.min(),x.max(),300)
    obj_smooth=spline(x,obj,x_new)
    '''
    plt.scatter(x, obj, c='black')
    plt.plot(x, obj, linewidth=1)
    #plt.plot(x_new,obj_smooth,c='red')
    plt.xlabel('steps')
    plt.ylabel('obj')
    plt.show()
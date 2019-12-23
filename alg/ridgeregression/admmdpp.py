import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from alg.utils import stop
from alg.utils import draw
#from scipy.interpolate import spline

def ADMM(A, b, alpha=0.1, beta=0.01, show_x=True, show_graph=True, log_int=1):
    #ADMM(x, alpha, landa, A, b, beta, z, D):
    n = A.shape[1]
    x = np.random.rand(n, 1)
    z = np.random.rand(n, 1)
    landa = np.random.rand(n, 1)
    D = np.identity(len(A[0]))
    k = 0
    if show_x:
        print("x starts with: ")
        print(x)
    red = np.linalg.norm(
            x, ord=1) + (alpha / 2) * np.linalg.norm(A @ x - b)**2
    print('The initial target value is %f' % (red))
    print("x的取值的迭代过程：")
    print(x)
    obj = []
    I = np.identity(len(A[0]))
    zero=np.zeros(len(b))
    zero=zero.T
    zeros=np.zeros((len(D[0]),len(D[0])))

    while(1):
        temp1=np.linalg.inv(alpha * (A.T @ A) + D)
        temp2=np.linalg.inv(alpha * (A.T @ b) + (D @ z) - landa)
        xx=temp1 @ temp2

        temp1=np.linalg.inv(2*I + D)
        temp2=landa + D @ xx
        zz=temp1 @ temp2

        landalanda=landa - D @ (xx-zz)
        red = np.linalg.norm(
            xx, ord=1) + (alpha / 2) * np.linalg.norm(A @ xx - b)**2
        if k % log_int == 0:
            obj.append(red)
            print('The %dth iteration, target value is %f' % (k, red))

        if stop(I,xx,-I,zz,z,zero,landalanda,alpha):
            break
        else:
            r = xx - zz
            s = -1 * alpha * (zz - z)
            u = 10
            t_incr = 2
            t_decr = t_incr
            if np.linalg.norm(r, ord=2) > u * np.linalg.norm(s, ord=2):
                print('penalty parameter increase')
                D = D * t_incr
            elif np.linalg.norm(s, ord=2) > u * np.linalg.norm(r, ord=2):
                print('penalty parameter decrease')
                D = D / t_decr
            x = np.copy(xx)
            z = np.copy(zz)
            landa = np.copy(landalanda)
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


main()

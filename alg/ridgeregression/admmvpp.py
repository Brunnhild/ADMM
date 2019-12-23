#Ridge Regression

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
#from scipy.interpolate import spline


def ADMM_VPP(A, b, alpha=0.1, beta=0.01, show_x=True, show_graph=True, log_int=1, show_penalty=True):
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
        part_1 = np.linalg.inv(alpha * (A.T @ A) + beta * I)
        part_2 = alpha * (A.T @ b) + beta * z - landa
        xx = part_1 @ part_2
        zz = (landa + beta * xx) / (2 + beta)
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
        if stop(I,xx,-I,zz,z,zero,landalanda,alpha):
            break
        else:
            r = xx - zz
            s = -1 * alpha * (zz - z)
            u = 10
            t_incr = 2
            t_decr = t_incr
            if np.linalg.norm(r, ord=2) > u * np.linalg.norm(s, ord=2):
                if show_penalty:
                    print('penalty parameter increase')
                beta = beta * t_incr
            elif np.linalg.norm(s, ord=2) > u * np.linalg.norm(r, ord=2):
                if show_penalty:
                    print('penalty parameter decrease')
                beta = beta / t_decr
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

def stop(A,xx,B,zz,z,b,landalanda,alpha):
    rr=A @ xx + B * zz - b
    ss=alpha * (A.T @ B @ (zz-z))
    n=xx.shape[0]
    epr=0.001
    epa=1
    epp=sqrt(n)*epa + epr * max(np.linalg.norm(A @ xx), np.linalg.norm(B @ zz), np.linalg.norm(b))
    epd=sqrt(n)*epa + epr * np.linalg.norm(A.T @ landalanda)
    if np.linalg.norm(rr,ord=2) <= epp and np.linalg.norm(ss,ord=2) <= epd:
        return True
    else:
        return False

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
    '''
    x = np.array([[1.]])
    alpha=0.1
    landa=0.5
    A = np.array([[1.]])
    b = np.array([[.2]])
    error=0.0001
    xx,count,obj=SD(x,alpha,landa,A,b,error)
    '''
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

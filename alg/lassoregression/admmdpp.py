import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
#from scipy.interpolate import spline

def ADMM(x,alpha,landa,A,b,beta,z,D):
    k = 0
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
        zz=np.sign(xx + np.linalg.inv(D) @ landa) * np.maximum(np.abs(xx + np.linalg.inv(D) @ landa) - np.linalg.inv(D),zeros)
        landalanda=landa + D @ (xx-zz)
        
        red = np.linalg.norm(
            xx, ord=1) + (alpha / 2) * np.linalg.norm(A @ xx - b)**2
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
            
    return xx, k, obj

def stop(A,xx,B,zz,z,b,landalanda,alpha):
    rr=A @ xx + B * zz - b
    ss=alpha * (A.T @ B @ (zz-z))
    n=xx.shape[0]
    epr=0.001
    epa=1
    epp=sqrt(n)*epa + epr * max(np.linalg.norm(A @ xx), np.linalg.norm(B @ zz), np.linalg.norm(b))
    epd=sqrt(n)*epa + epr * np.linalg.norm(A.T @ landalanda)
    if np.linalg.norm(rr) <= epp and np.linalg.norm(ss) <= epd:
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
    x = np.array([[1.]])
    z = np.array([[1.]])
    landa = np.array([[1.]])
    A = np.array([[1.]])
    b = np.array([[.2]])
    alpha=0.1
    beta=0.01
    D = np.array([[0.5]])
    xx,count,obj=ADMM(x,alpha,landa,A,b,beta,z,D)

    draw(obj)
    print("最终迭代结果: x：", xx)
    print("共进行了", count, "次迭代")


main()

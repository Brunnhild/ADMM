#LASSO Regression

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
#from scipy.interpolate import spline


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

def Lasso_ADMM_Nestersr(alpha,landa,A,b,beta,yx,yz):
    k=0
    nl = 0
    x = yx
    z = yz
    print("x的取值的迭代过程：")
    print(x)
    obj=[]
    I = np.identity(len(A[0]))
    zero = np.zeros(len(b))
    zero = zero.T
    while(1):
        nl_new = (1 + sqrt(1+4*nl*nl)) / 2
        gamma = (1-nl)/nl_new
        temp1 = np.linalg.inv(alpha*(A.T@A)+beta*I)
        temp2 = alpha * (A.T @ b) + beta * z - landa
        y_xx = temp1 @ temp2
        xx = (1-gamma)*y_xx + gamma*yx
        y_zz = np.sign(xx+landa / beta) * np.maximum(np.abs(xx+landa/beta)-1/beta,0)
        zz = (1-gamma)*y_zz + gamma*yz
        landalanda = landa + beta*(xx - zz)
        red = np.linalg.norm(
            xx, ord=1) + (alpha / 2) * np.linalg.norm(A @ xx - b) ** 2
        obj.append(red)
        print('The %dth iteration, target value is %f' % (k, red))
        #A,xx,B,zz,z,b,landalanda,alpha
        if stop(I,xx,-I,zz,z,zero,landalanda,alpha):
            break
        else:
            x = xx
            z = zz
            yx = y_xx
            yz = y_zz
            landa = landalanda
            nl = nl_new
            print('The current x is: ', x)
            k = k + 1
    return xx,k,obj

def draw(obj):
    x=np.zeros(len(obj))
    for i in range(len(obj)):
        x[i]=i*0.1
    '''
    x_new=np.linspace(x.min(),x.max(),300)
    obj_smooth=spline(x,obj,x_new)
    '''
    plt.scatter(x,obj,c='black')
    plt.plot(x,obj,linewidth=1)
    #plt.plot(x_new,obj_smooth,c='red')
    plt.ylabel('obj')
    plt.show()

def main():

    yx = np.array([[1.]])
    yz = np.array([[1.]])
    landa = np.array([[1.]])
    A = np.array([[1.]])
    b = np.array([[.2]])
    alpha=0.1
    error=0.01
    beta=0.01
    #alpha,landa,A,b,beta,yx,yz
    xx,count,obj=Lasso_ADMM_Nestersr(alpha,landa,A,b,beta,yx,yz)
    draw(obj)

    print("最终迭代结果: x：",xx)
    print("共进行了",count,"次迭代")

if __name__ == '__main__':
    main()

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def LASSO_ADMM(x, alpha, landa, A, b, error, beta, z):
    k = 0
    print("x的取值的迭代过程：")
    print(x)
    obj = []
    while (1):
        I = np.identity(len(A[0]))
        #此处的landa是向量
        temp1 = np.linalg.inv(alpha * (A.T @ A) + (beta + 1.0 / beta) * I)

        temp2 = alpha * (A.T @ b) + beta * z - landa + (1.0 / beta) * x

        xx = temp1 @ temp2

        zz = np.copy(z)

        zz = np.sign(xx + landa / beta) * np.maximum(
            np.abs(xx + landa / beta) - 1 / beta, 0)

        landalanda = landa + beta * np.abs(xx - zz)

        red = np.linalg.norm(
            xx, ord=1) + (alpha / 2) * np.linalg.norm(A @ xx - b)**2
        obj.append(red)
        print('The %dth iteration, target value is %f' % (k, red))

        xzk = np.vstack((x, z))
        xzkp = np.vstack((xx, zz))

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


def RidgeRegression_ADMM(x, alpha, landa, A, b, error, beta, z):
    k = 0
    print("x的取值的迭代过程：")
    print(x)
    obj = []
    while (1):
        I = np.identity(len(A[0]))
        part_1 = np.linalg.inv(alpha * (A.T @ A) + (beta + 1.0 / beta) * I)

        part_2 = alpha * (A.T @ b) + beta * z - landa + (1.0 / beta) * x

        xx = part_1 @ part_2

        zz = (landa + beta * xx) / (2 + beta)

        landalanda = landa - beta * np.abs(xx - zz)

        red = np.linalg.norm(
            xx, ord=1) + (alpha / 2) * np.linalg.norm(A @ xx - b)**2

        obj.append(red)

        print('The %dth iteration, target value is %f' % (k, red))

        xzk = np.vstack((x, z))
        xzkp = np.vstack((xx, zz))

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
    plt.plot(x, obj, linewidth=0.5)
    #plt.plot(x_new,obj_smooth,c='red')
    plt.ylabel('obj')
    plt.show()


def main():
    #LASSO-ADMM
    '''
    x = np.array([[1.]])
    z = np.array([[1.]])
    landa = np.array([[1.]])
    A = np.array([[1.]])
    b = np.array([[.2]])
    alpha=0.1
    error=0.01
    beta=0.01
    xx,count,obj=LASSO_ADMM(x,alpha,landa,A,b,error,beta,z)
    '''
    #RidgeRegression_ADMM

    x = np.array([[1.]])
    z = np.array([[1.]])
    landa = np.array([[1.]])
    A = np.array([[1.]])
    b = np.array([[.2]])
    alpha = 0.1
    error = 0.01
    beta = 0.01
    xx, count, obj = RidgeRegression_ADMM(x, alpha, landa, A, b, error, beta,
                                          z)

    draw(obj)
    print("最终迭代结果: x：", xx)
    print("共进行了", count, "次迭代")


main()

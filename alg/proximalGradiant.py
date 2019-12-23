import numpy as np


#近端梯度法
def ProximalGradiant(x, alpha, landa, A, b):
    k = 0
    count = 0
    print("x的取值的迭代过程：")
    print(x)
    while (1):
        count += 1
        temp = np.dot(A, x) - b
        z = x - alpha * landa * np.dot(A.T, temp)
        xx = np.zeros(len(z))
        xx = xx.T
        for i in range(len(z)):
            xx[i] = np.sign(z[i]) * max(abs(z[i]) - alpha, 0)
        if (xx == x).all():
            break
        else:
            x = np.copy(xx)
            print(x)
    return xx, count


def main():
    x = np.array([0.1, 0.2, 0.3])
    x = x.T
    alpha = 0.1
    landa = 0.5
    A = np.array([[1, 0.2, 3], [2, 0.1, 3], [0.3, 1, 2]])
    b = np.array([1, 0.3, 2])
    b = b.T
    xx, count = ProximalGradiant(x, alpha, landa, A, b)
    print("最终迭代结果: x：", xx)
    print("共进行了", count, "次迭代")


main()

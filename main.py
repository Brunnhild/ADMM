from data.breastcancer import get_data
from alg.lassoregression.admmvpp import ADMM


if __name__ == '__main__':
    A, b = get_data()
    ADMM(A, b, show_x=False, log_int=100)
    # print(b.shape)

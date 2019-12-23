from data.multidim import get_data
from alg.ridgeregression.admmnes import ADMM
from alg.lassoregression.PGM import PGM
from alg.ridgeregression.SD import SD
from alg.ridgeregression.BFGS import BFGS


if __name__ == '__main__':
    A, b = get_data()
    ADMM(A, b, show_x=False, log_int=1)
    # PGM(A, b, show_x=False, log_int=1)
    # SD(A, b, show_x=False, log_int=1)
    # BFGS(A, b, show_x=False, log_int=10)
    # print(b.shape)

import numpy as np
from data.breastcancer import get_data
from alg.ridgeregression.admmnes import ADMM
from alg.lassoregression.PGM import PGM
from alg.ridgeregression.SD import SD
from alg.ridgeregression.BFGS import BFGS


if __name__ == '__main__':
    np.random.seed(2)
    A, b = get_data()
    ADMM(A, b, show_x=False, log_int=1, show_graph=True, max_step=-1)
    # PGM(A, b, show_x=False, log_int=1)
    # SD(A, b, show_x=False, log_int=1)
    # BFGS(A, b, show_x=False, log_int=10)
    # print(b.shape)

import pandas as pd
import numpy as np
from scipy.stats import anderson_ksamp
from hyppo.ksample import Energy, MMD, DISCO
from scipy.stats import laplace, norm, ks_2samp, cramervonmises_2samp


chat_id = 682673597 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    return MMD(compute_kernel="laplacian", gamma=1).test(x, y)[1] < 0.08

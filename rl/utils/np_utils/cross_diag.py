import numpy as np


def cross_diag(v: np.ndarray):
    n = v.shape[0]
    i = np.arange(n)
    re = np.zeros((n, n), dtype=v.dtype)
    re[i, -i-1] = v
    return re

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def calc_inv(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    assert x.shape == y.shape == w.shape
    assert len(x.shape) == 1 or x.shape[0] == 1 or x.shape[1] == 1

    w_ = scipy.sparse.diags(w)

    x_ = np.zeros(shape=(x.size, 2))
    x_[:, 0] = 1
    x_[:, 1] = x

    x_ = scipy.sparse.csc_matrix(x_)

    x_tr = x_.transpose()

    p1 = x_tr.dot(w_).dot(x_)
    p2 = scipy.sparse.linalg.inv(p1)
    res = p2.dot(x_tr).dot(w_).dot(y)

    return res

import pandas as pd
import numpy as np


def mean_squared_error(x: pd.Series, y: pd.Series) -> float:
    diff: np.ndarray = y - x
    sq_diff: np.ndarray = diff ** 2
    res = sq_diff.mean()

    return res


def mean_percent_error(x: pd.Series, y: pd.Series) -> float:
    mpe_p = (y - x) / x
    mpe_p = mpe_p[mpe_p.isna() == False]
    mpe = np.average(mpe_p) * 100

    return mpe


def r2_score(x: pd.Series, y: pd.Series) -> float:
    from sklearn.metrics import r2_score as original_r2
    valid_ind = (x.isna() == False) & (y.isna() == False)
    x_ = x[valid_ind == True]
    y_ = y[valid_ind == True]

    ret = original_r2(x_,y_)
    return ret

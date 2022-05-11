import numpy as np
import pandas as pd
import sklearn.metrics

__all__ = ['rmse', 'mpe', 'correlation', 'r_2']


def rmse(y_true: pd.Series, y_pred: pd.Series):
    return np.sqrt(sklearn.metrics.mean_squared_error(y_true,y_pred))


def mpe(y_true: pd.Series, y_pred: pd.Series):
    min_value: float = 0.1
    filtered = y_true.copy()
    filtered[y_true.eq(0)] = min_value
    diff = y_pred - y_true
    p_diff = (diff / filtered) * 100
    res = np.mean(p_diff)
    return res


def correlation(y_true: pd.Series, y_pred: pd.Series):
    return np.sqrt(sklearn.metrics.r2_score(y_true,y_pred))


def r_2(y_true: pd.Series, y_pred: pd.Series):
    return sklearn.metrics.r2_score(y_true,y_pred)
import tensorflow.keras.metrics
import tensorflow_probability as tfp
from tensorflow.keras import backend as K

__all__ = ['rmse', 'mpe', 'correlation', 'r_2','diff_std']


def rmse(y_true, y_pred):
    return K.sqrt(tensorflow.keras.metrics.mse(y_true,y_pred))


def mpe(y_true, y_pred):
    min_value: float = 0.1
    filtered = K.switch(K.equal(y_true, 0), K.zeros_like(y_true) + min_value, y_true)
    diff = y_pred - y_true
    p_diff = (diff / filtered) * 100
    res = K.mean(p_diff)
    return res


def correlation(y_true, y_pred):
    return tfp.stats.correlation(K.expand_dims(y_true), K.expand_dims(y_pred))


def r_2(y_true, y_pred):
    return tfp.stats.correlation(K.expand_dims(y_true), K.expand_dims(y_pred)) ** 2

def diff_std(y_true, y_pred):
    return K.std(y_true - y_pred)
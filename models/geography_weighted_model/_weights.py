import numpy as np
import typing

_T = typing.TypeVar('_T')


def calc_2D_weights(x: _T, y: _T, center: typing.Tuple[float, float], range: _T) -> _T:
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    in_range = dist < range

    ret = x * 0
    ret[in_range] = np.abs(1 - (dist[in_range] / range)**2) ** 2

    return ret

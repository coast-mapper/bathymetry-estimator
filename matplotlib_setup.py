import matplotlib
import typing

matplotlib.use('Agg')

from matplotlib import colors

from matplotlib import pyplot as plt

__all__ = ["plt", "colors", "set_figure_size", "get_figure_size"]

__figsize: typing.Optional[typing.Tuple[float, float]] = None


def set_figure_size(figsize: typing.Optional[typing.Tuple[float, float]]):
    global __figsize
    __figsize = figsize


def get_figure_size() -> typing.Optional[typing.Tuple[float, float]]:
    global __figsize
    return __figsize

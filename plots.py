import numpy as np
import pandas as pd
import typing

from matplotlib_setup import plt, colors,get_figure_size


def fit_scatter(y_true: pd.Series, y_pred: pd.Series, true_label: str, pred_label: str, title: typing.Optional[str],
                without_description: bool = False):
    import matplotlib.colorbar
    # reference_stats = y_true.describe(percentiles=[.99])
    # prediction_stats = y_pred.describe(percentiles=[.99])

    maximal = 16  # max(reference_stats['99%'], prediction_stats['99%']) * 1.05

    h, xe, ye = np.histogram2d(y_true, y_pred, bins=[50, 50], range=[[0, maximal], [0, maximal]])

    sh = np.sum(h)
    nh: np.ndarray = (h / sh) * 100

    xx_: np.ndarray
    yy_: np.ndarray

    xx_, yy_ = np.meshgrid(xe[:-1], ye[:-1])

    xx_flat = xx_.flatten('F')
    yy_flat = yy_.flatten('F')

    nh_flat = nh.flatten()

    plt.figure(figsize=get_figure_size())
    plt.hist2d(xx_flat, yy_flat, bins=[xe, ye], weights=nh_flat, norm=colors.LogNorm(vmin=1e-4, vmax=10))

    if not without_description:
        cbar: matplotlib.colorbar.Colorbar = plt.colorbar()
        cbar.set_label('Frequency [%]', rotation=270, labelpad=12)
        plt.xlabel(true_label)
        plt.ylabel(pred_label)

    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, maximal])
    plt.xticks(np.arange(0,17,4))
    plt.ylim([0, maximal])
    plt.yticks(np.arange(0,17,4))
    plt.plot([0, maximal], [0, maximal])

    if not without_description and title is not None:
        plt.title(title)

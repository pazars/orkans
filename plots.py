import sys
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
from pysteps.utils import transformation

try:
    from orkans.preprocessing import find_best_boxcox_lambda
except ModuleNotFoundError:
    LIB_SRC_DIR = Path("C:/Users/davis.pazars/Documents").as_posix()
    sys.path.append(LIB_SRC_DIR)
finally:
    from orkans.preprocessing import find_best_boxcox_lambda


def _plot_distribution(data, labels, skw):

    # visualize the data distribution with boxplots and plot the
    # corresponding skewness

    N = len(data)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax2.plot(np.arange(N + 2), np.zeros(N + 2), ":r")
    ax1.boxplot(data, labels=labels, sym="", medianprops={"color": "k"})

    ymax = []
    for i in range(N):
        y = skw[i]
        x = i + 1
        ax2.plot(x, y, "*r", ms=10, markeredgecolor="k")
        ymax.append(np.max(data[i]))

    # ylims
    ylims = np.percentile(ymax, 50)
    ax1.set_ylim((-1 * ylims, ylims))
    ylims = np.max(np.abs(skw))
    ax2.set_ylim((-1.1 * ylims, 1.1 * ylims))

    # labels
    ax1.set_ylabel(r"Standardized values [$\sigma$]")
    ax2.set_ylabel(r"Skewness []", color="r")
    ax2.tick_params(axis="y", labelcolor="r")


def compare_transformations(rainrate, metadata):

    data = []
    labels = []
    skw = []

    rainrate_flat = rainrate[rainrate > metadata["zerovalue"]].flatten()

    data.append((rainrate_flat - np.mean(rainrate_flat)) / np.std(rainrate_flat))
    labels.append("Original")
    skw.append(stats.skew(rainrate_flat))

    rrate_, _ = transformation.dB_transform(rainrate_flat, metadata)
    data.append((rrate_ - np.mean(rrate_)) / np.std(rrate_))
    labels.append("dB")
    skw.append(stats.skew(rrate_))

    rrate_, _ = transformation.sqrt_transform(rainrate_flat, metadata)
    data.append((rrate_ - np.mean(rrate_)) / np.std(rrate_))
    labels.append("sqrt")
    skw.append(stats.skew(rrate_))

    Lambda = find_best_boxcox_lambda(rainrate_flat, metadata)
    rrate_, _ = transformation.boxcox_transform(rainrate_flat, metadata, Lambda)
    data.append((rrate_ - np.mean(rrate_)) / np.std(rrate_))
    labels.append("Box-Cox")
    skw.append(stats.skew(rrate_))

    rrate_, _ = transformation.NQ_transform(rainrate_flat, metadata)
    data.append((rrate_ - np.mean(rrate_)) / np.std(rrate_))
    labels.append("NQ")
    skw.append(stats.skew(rrate_))

    _plot_distribution(data, labels, skw)
    plt.title("Data transforms")
    plt.tight_layout()

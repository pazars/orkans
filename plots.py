import os
import sys
import shutil
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
from pysteps import verification
from pysteps.utils import transformation
from pysteps.postprocessing import ensemblestats
from pysteps.visualization import plot_precip_field

try:
    from orkans.preprocessing import find_best_boxcox_lambda
except ModuleNotFoundError:
    LIB_DIR = (Path(".") / "..").resolve().as_posix()
    sys.path.append(LIB_DIR)
finally:
    from orkans.preprocessing import find_best_boxcox_lambda
    from orkans import PLOT_DIR


class PlotProcessor:
    """Class for saving plots."""

    def __init__(self, rid, ref_data, data, metadata) -> None:
        self.run_id = rid
        self.ref_data = ref_data
        self.data = data
        self.metadata = metadata
        self.plot_dir = self._create_plot_dir()

    def _create_plot_dir(self):
        """Create a directory for plots of a specific run.

        Format is ./results/plots/RUN-ID
        """
        plot_dir = PLOT_DIR / self.run_id

        if plot_dir.exists():
            shutil.rmtree(plot_dir)
        else:
            plot_dir.mkdir(parents=True)

        return plot_dir

    def _calculate_leadtime(self, lead_idx: int):
        tstep = self.metadata["accutime"]
        if lead_idx >= 0:
            return tstep * (lead_idx + 1)
        else:
            return tstep * (self.data.shape[0] - lead_idx + 1)

    def save_rank_histogram(self, lead_idx: int):
        rankhist = verification.rankhist_init(self.data.shape[0], 0.1)
        nowcast = self.data[:, lead_idx, :, :]
        reference = self.ref_data[lead_idx, :, :]
        verification.rankhist_accum(rankhist, nowcast, reference)

        _, ax = plt.subplots()
        verification.plot_rankhist(rankhist, ax)

        leadtime = int(self._calculate_leadtime(lead_idx))
        ax.set_title(f"Rank histogram (+{leadtime} min)")
        fname = f"rank_histogram_T{leadtime}.svg"
        plt.savefig(self.plot_dir / fname, format="svg")

    def save_reliability_diagram(self, lead_idx: int):
        reldiag = verification.reldiag_init(0.1)
        nowcast = self.data[:, lead_idx, :, :]
        reference = self.ref_data[lead_idx, :, :]
        exc_probs = ensemblestats.excprob(nowcast, 0.1, ignore_nan=True)
        verification.reldiag_accum(reldiag, exc_probs, reference)

        _, ax = plt.subplots()
        verification.plot_reldiag(reldiag, ax)
        leadtime = int(self._calculate_leadtime(lead_idx))
        ax.set_title(f"Reliability diagram (T+{leadtime}min)")
        fname = f"reliability_diagram_T{leadtime}.svg"
        plt.savefig(self.plot_dir / fname, format="svg")

    def save_roc_curve(self, lead_idx: int):
        roc = verification.ROC_curve_init(0.1, n_prob_thrs=10)
        nowcast = self.data[:, lead_idx, :, :]
        reference = self.ref_data[lead_idx, :, :]
        exc_probs = ensemblestats.excprob(nowcast, 0.1, ignore_nan=True)
        verification.ROC_curve_accum(roc, exc_probs, reference)

        _, ax = plt.subplots()
        verification.plot_ROC(roc, ax, opt_prob_thr=True)

        leadtime = self._calculate_leadtime(lead_idx)
        ax.set_title(f"Rank histogram (+{leadtime} min)")
        plt.savefig(self.plot_dir / "roc_curve.svg", format="svg")

    def _plot_distribution(self, data, labels, skw):

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

    def save_transform_comparison(self):

        rainrate = self.ref_data
        metadata = self.metadata

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

        self._plot_distribution(data, labels, skw)
        plt.title("Data transforms")
        plt.tight_layout()
        plt.savefig(self.plot_dir / "data_transform_comparison.svg", format="svg")

    def save_last_precip_field(self):
        """pysteps plot_precip_field wrapper for last observed timestep."""
        plot_precip_field(self.ref_data[-1, :, :], geodata=self.metadata)
        plt.savefig(self.plot_dir / "last_obs_precip_field.svg", format="svg")

    def save_nowcast_field(self, lead_idx: int, ensemble=False):
        """pysteps plot_precip_field wrapper for nowcast timesteps."""
        leadtime = int(self._calculate_leadtime(lead_idx))
        title = f"T+{leadtime}"

        if ensemble:
            ensemble_mean = np.mean(self.data[:, lead_idx, :, :], axis=0)
            plot_precip_field(ensemble_mean, geodata=self.metadata, title=title)
        else:
            plot_precip_field(
                self.data[lead_idx, :, :], geodata=self.metadata, title=title
            )

        fname = f"nowcast_precip_field_T{leadtime}min.svg"
        plt.savefig(self.plot_dir / fname, format="svg")

    def save_all_ensemble_plots(self, lead_idx: int):

        self.save_transform_comparison()
        plt.clf()
        self.save_last_precip_field()
        plt.clf()
        self.save_nowcast_field(lead_idx, ensemble=True)
        plt.clf()
        self.save_rank_histogram(lead_idx)
        plt.clf()
        self.save_reliability_diagram(lead_idx)
        plt.clf()
        self.save_roc_curve(lead_idx)

    def save_all_deterministic_plots(self, lead_idx: int):

        self.save_transform_comparison()
        plt.clf()
        self.save_last_precip_field()
        plt.clf()
        self.save_nowcast_field(lead_idx)

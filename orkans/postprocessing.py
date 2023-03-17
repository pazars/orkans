import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from pysteps import verification
from pysteps.postprocessing import ensemblestats
from pysteps.utils import transformation
from pysteps.verification import detcontscores, ensscores
from pysteps.visualization import plot_precip_field
from scipy import stats

from orkans import OUT_DIR, PLOT_DIR
from orkans.preprocessing import find_best_boxcox_lambda


class PostProcessor:
    """Class for nowcast skill evaluation."""

    def __init__(self, rid, obs, pred, metadata) -> None:
        self.obs = obs
        self.pred = pred
        self.metadata = metadata
        self.is_ensemble = len(pred.shape) > 3

        self.plots = PlotProcessor(rid, obs, pred, metadata)

    def calc_scores(self, threshold: float, cfg: dict, lead_idx: int) -> dict:

        if self.is_ensemble:
            return self.calc_ens_scores(cfg, lead_idx, thr=threshold)
        else:
            return self.calc_det_scores(cfg, lead_idx, thr=threshold)

    def calc_det_scores(self, cfg: dict, lead_idx: int, thr: float = 0.1) -> dict:

        pred = self.pred[lead_idx, :, :]
        obs = self.obs[lead_idx, :, :]

        res = {"nwc_type": "deterministic"}
        metrics = cfg["metrics"]["deterministic"]
        score_map = detcontscores.det_cont_fct(pred, obs, metrics, thr=thr)

        leadtime = self.plots._calculate_leadtime(lead_idx)
        for metric, score in score_map.items():
            new_metric_name = f"{metric}_T{int(leadtime)}_THR{thr}"
            res[new_metric_name] = score

        return res

    def calc_ens_scores(self, cfg: dict, lead_idx: int, thr: float = 0.1) -> dict:

        pred = self.pred[:, lead_idx, :, :]
        obs = self.obs[lead_idx, :, :]

        res = {"nwc_type": "ensemble"}
        mean_metrics = cfg["metrics"]["ensemble"]["mean"]

        leadtime = self.plots._calculate_leadtime(lead_idx)
        for metric in mean_metrics:
            score = ensscores.ensemble_skill(pred, obs, metric, thr=thr)
            new_metric_name = f"{metric}_T{int(leadtime)}_THR{thr}"
            res[new_metric_name] = score

        # Compute area under ROC
        roc_metric_name = f"roc_area_T{int(leadtime)}_THR{thr}"
        res[roc_metric_name] = self.plots.roc_curve(lead_idx, thr, area=True)

        return res

    def save_plots(self, lead_idx: int = 0) -> None:
        if self.is_ensemble:
            self.plots.save_all_ensemble_plots(lead_idx)
        else:
            self.plots.save_all_deterministic_plots(lead_idx)

    def save_results(self, model_name, out_data):

        if not out_data:
            logger.error("Nowcast didn't return anything. Exiting.")
            exit()

        fname = f"nowcasts_{model_name}.csv"

        if not OUT_DIR.exists():
            os.mkdir(OUT_DIR)

        out_path = OUT_DIR / fname

        if out_path.exists():
            data = pd.read_csv(out_path)
        else:
            data = pd.DataFrame()

        new_data = pd.DataFrame.from_dict([out_data])

        # Output run results
        data = pd.concat([data, new_data])

        # Index set to True leads to a redundant column at next read
        data.to_csv(out_path, index=False)


class PlotProcessor:
    """Class for saving plots.
    Intended for use in PostProcessor, but can be used on its own.
    """

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
        plot_dir.mkdir(parents=True)

        return plot_dir

    def _calculate_leadtime(self, lead_idx: int):
        tstep = self.metadata["accutime"]
        if lead_idx >= 0:
            return tstep * (lead_idx + 1)
        else:
            return tstep * (self.data.shape[0] - lead_idx + 1)

    def rank_histogram(self, lead_idx: int, thr: float = 0.1):
        rankhist = verification.rankhist_init(self.data.shape[0], thr)
        nowcast = self.data[:, lead_idx, :, :]
        reference = self.ref_data[lead_idx, :, :]
        verification.rankhist_accum(rankhist, nowcast, reference)

        _, ax = plt.subplots()
        verification.plot_rankhist(rankhist, ax)

        leadtime = int(self._calculate_leadtime(lead_idx))
        ax.set_title(f"Rank histogram (+{leadtime} min)")
        thr_parts = str(thr).split(".")
        fname = f"rank-histogram-T{leadtime}-thr{thr_parts[0]}_{thr_parts[1]}.svg"
        plt.savefig(self.plot_dir / fname, format="svg")

    def reliability_diagram(self, lead_idx: int, thr: float = 0.1):
        reldiag = verification.reldiag_init(thr)
        nowcast = self.data[:, lead_idx, :, :]
        reference = self.ref_data[lead_idx, :, :]
        exc_probs = ensemblestats.excprob(nowcast, thr, ignore_nan=True)
        verification.reldiag_accum(reldiag, exc_probs, reference)

        _, ax = plt.subplots()
        verification.plot_reldiag(reldiag, ax)
        leadtime = int(self._calculate_leadtime(lead_idx))
        ax.set_title(f"Reliability diagram (T+{leadtime}min)")
        thr_parts = str(thr).split(".")
        fname = f"reliability-diagram-T{leadtime}-thr{thr_parts[0]}_{thr_parts[1]}.svg"
        plt.savefig(self.plot_dir / fname, format="svg")

    def roc_curve(self, lead_idx: int, thr: float = 0.1, area: bool = False):
        roc = verification.ROC_curve_init(thr, n_prob_thrs=10)
        nowcast = self.data[:, lead_idx, :, :]
        reference = self.ref_data[lead_idx, :, :]
        exc_probs = ensemblestats.excprob(nowcast, thr, ignore_nan=True)
        verification.ROC_curve_accum(roc, exc_probs, reference)

        if area:
            _, _, roc_area = verification.probscores.ROC_curve_compute(roc, True)
            return roc_area

        _, ax = plt.subplots()
        verification.plot_ROC(roc, ax, opt_prob_thr=True)

        leadtime = int(self._calculate_leadtime(lead_idx))
        ax.set_title(f"ROC; T+{leadtime} min; threshold={thr}")
        thr_parts = str(thr).split(".")
        fname = f"roc-T{leadtime}-thr{thr_parts[0]}_{thr_parts[1]}.svg"
        plt.savefig(self.plot_dir / fname, format="svg")

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
        self.rank_histogram(lead_idx)
        plt.clf()
        self.reliability_diagram(lead_idx)
        plt.clf()
        self.roc_curve(lead_idx)

    def save_all_deterministic_plots(self, lead_idx: int):

        self.save_transform_comparison()
        plt.clf()
        self.save_last_precip_field()
        plt.clf()
        self.save_nowcast_field(lead_idx)

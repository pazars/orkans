import datetime
import yaml

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from pathlib import Path
from scipy import stats
from loguru import logger
from pathlib import Path
from pysteps import io, nowcasts, rcparams
from pysteps.utils import conversion, transformation, clip_domain
from pysteps.visualization import plot_precip_field
from pysteps.motion.lucaskanade import dense_lucaskanade

from preprocessing import PreProcessor

# Log file path for loguru
_LOG_PATH = (Path(".") / "logs" / "log_{time}.txt").resolve()

# Precipitation ratio threshold
_PRECIP_RATIO_THR = 0.25


def load_config(cfg_path: Path = None):

    # Load organisation specific pysteps configuration file
    # It does not replace .pystepsrc!
    if not cfg_path:
        script_dir = Path(".").resolve()
        cfg_path = script_dir / "config.yaml"

    with open(cfg_path, "r") as file:
        return yaml.safe_load(file)


def load_rainrate_data(cfg: dict):

    gen_cfg = cfg["general"]

    data_source = gen_cfg["data_source"]
    start_time = str(gen_cfg["start_time"])
    start_fmt = str(gen_cfg["start_fmt"])
    date = datetime.strptime(start_time, start_fmt)

    # Load data source config
    root_path = rcparams.data_sources[data_source]["root_path"]
    path_fmt = rcparams.data_sources[data_source]["path_fmt"]
    fn_pattern = rcparams.data_sources[data_source]["fn_pattern"]
    fn_ext = rcparams.data_sources[data_source]["fn_ext"]
    importer_name = rcparams.data_sources[data_source]["importer"]
    importer_kwargs = rcparams.data_sources[data_source]["importer_kwargs"]
    timestep = rcparams.data_sources[data_source]["timestep"]
    n_flow_steps = gen_cfg["n_flow_steps"]

    n_leadtimes = 0
    # Load forecast reference data (reanalysis)
    if gen_cfg["verify"]:
        n_leadtimes = gen_cfg["n_leadtimes"]

    # Find the radar files in the archive
    filenames = io.find_by_date(
        date,
        root_path,
        path_fmt,
        fn_pattern,
        fn_ext,
        timestep,
        num_prev_files=n_flow_steps - 1,
        num_next_files=n_leadtimes,
    )

    # Read the data from the archive
    importer = io.get_method(importer_name, "importer")
    series_data = io.read_timeseries(filenames, importer, **importer_kwargs)
    data, quality, metadata = series_data

    # Data should already be in mm/h, but convert just in case it isn't
    rainrate, metadata = conversion.to_rainrate(data, metadata)

    # Clip domain to a specific region
    # If first entry is None, uses whole domain
    domain_box = gen_cfg["domain_box"]
    if domain_box[0]:
        # List values parsed as strings, so need to convert back to floats
        domain_box = [float(num) for num in domain_box]
        rainrate, metadata = clip_domain(rainrate, metadata, domain_box)

    return (rainrate, metadata)


# Define method to visualize the data distribution with boxplots and plot the
# corresponding skewness
def _plot_distribution(data, labels, skw):

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


def find_best_boxcox_lambda(rainrate, metadata, plot: bool = False, verbose: bool = False):

    data = []
    labels = []
    skw = []

    # Keep only positive rainfall values
    rainrate_flat = rainrate[rainrate > metadata["zerovalue"]].flatten()

    # Test a range of values for the transformation parameter Lambda
    lambdas = np.linspace(-0.4, 0.4, 11)
    for i, Lambda in enumerate(lambdas):
        R_, _ = transformation.boxcox_transform(rainrate_flat, metadata, Lambda)
        R_ = (R_ - np.mean(R_)) / np.std(R_)
        data.append(R_)
        labels.append("{0:.2f}".format(Lambda))
        skw.append(stats.skew(R_))  # skewness

    # Best lambda
    idx_best = np.argmin(np.abs(skw))

    if verbose:
        print("Best Box-Cox lambda:", lambdas[idx_best])
        print(f"Best Box-Cox skewness={skw[idx_best]} for lambda={lambdas[idx_best]}\n")

    # Plot the transformed data distribution as a function of lambda
    if plot:
        _plot_distribution(data, labels, skw)
        plt.title("Box-Cox transform")
        plt.tight_layout()
        plt.show()

    return lambdas[idx_best]


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


def find_best_transformation(rrate, mdata, verbose: bool = False):

    if verbose:
        print("-- Finding best data transformation --")

    skew_threshold = 0.2
    best_mean = 1e6
    best_skew = None

    best_transform = None
    extra_arg = None

    # Map of transformation methods
    transforms = {
        "dB": transformation.dB_transform,
        "sqrt": transformation.sqrt_transform,
        "box-cox": transformation.boxcox_transform,
        "nq": transformation.NQ_transform,
    }

    # Exclude zero values as most transformations can't deal with them
    rrate = rrate[rrate > mdata["zerovalue"]]

    # Reduce data array dimensionality to 1D for stats functions
    rrate_flat = rrate.flatten()

    for name, func in transforms.items():

        # Apply transformation. Extra step for Box-Cox
        if name != "box-cox":
            rrate_, _ = func(rrate_flat, mdata)
        else:
            Lambda = find_best_boxcox_lambda(rrate_flat, mdata, verbose=verbose)
            rrate_, _ = func(rrate_flat, mdata, Lambda)

        skewness = stats.skew(rrate_)

        # Skip if skewness doesn't meet threshold
        if skewness > skew_threshold:
            continue

        # Skip if no mean value improvement
        mean = np.mean(rrate_)
        if abs(mean) > abs(best_mean):
            continue

        # Update best performing method variables
        best_skew = skewness
        best_mean = mean
        best_transform = name

        if name == "box-cox":
            extra_arg = Lambda

    if verbose:
        print("Best transform method:", best_transform)
        print("Skewness:", best_skew)
        print(f"Mean: {best_mean}\n")

    # Apply best transformation to data with non-reduced dimensionality
    best_func = transforms[best_transform]
    # best_data, best_metadata = best_func(rrate, mdata)

    return (best_func, extra_arg)


def apply_transformation(rrate, mdata, tfunc, arg=None):

    # Even though the transformation was already applied in the find function,
    # it is applied again, because previously the data was flattened to 1D.
    # This approach avoids copying the data in the find function.
    if arg:
        return tfunc(rrate, mdata, Lambda=arg)
    return tfunc(rrate, mdata)


@logger.catch
def run():

    logger.add(
        _LOG_PATH,
        backtrace=True,
        diagnose=True,
        rotation="1 day",
        retention="1 week",
    )

    logger.info("Run started.")

    plot = False
    verbose = False

    # Load configuration file
    cfg = load_config()

    # Load rain rate data
    raw_rainrate, raw_metadata = load_rainrate_data(cfg)

    # PRE-PROCESSING

    pre_processor = PreProcessor()
    pre_processor.add_data(raw_rainrate, raw_metadata)

    # Perform nowcast only if more than a certain threshold of cells have precipitation
    ratio = pre_processor.precipitation_ratio(-1)
    if ratio < _PRECIP_RATIO_THR:
        msg = f"Precipitation ratio below threshold ({ratio} < {_PRECIP_RATIO_THR})."
        logger.warning(msg)
        logger.info("Run finished.")
        return

    out_data = pre_processor.collect_info()

    # Number of timesteps to use for velocity field estimation
    n_flow_steps = cfg["general"]["n_flow_steps"]

    if plot:
        # Plot the rainfall field
        plot_precip_field(raw_rainrate[n_flow_steps - 1, :, :], geodata=raw_metadata)
        plt.show()

    # Automatically transform data using most appropriate transformation
    best_tfunc, Lambda = find_best_transformation(raw_rainrate, raw_metadata, verbose=verbose)
    rainrate, metadata = apply_transformation(raw_rainrate, raw_metadata, best_tfunc, Lambda)

    if plot:
        compare_transformations(raw_rainrate, raw_metadata)
        plt.show()

    # Estimate the motion field
    vfield = dense_lucaskanade(rainrate[:n_flow_steps, :, :])

    # Run nowcast
    n_leadtimes = cfg["general"]["n_leadtimes"]
    model_names = ["steps"]
    for model_name in model_names:
        model_func = nowcasts.get_method(model_name)
        model_kwargs = cfg["model"][model_name]
        model_kwargs["precip_thr"] = 2
        rainrate_forecast = model_func(rainrate, vfield, n_leadtimes, **model_kwargs)


if __name__ == "__main__":

    run()

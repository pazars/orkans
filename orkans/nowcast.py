import time

from loguru import logger
from pysteps import nowcasts
from pysteps.motion.lucaskanade import dense_lucaskanade

from orkans import PRECIP_RATIO_THR
from orkans import utils
from orkans.postprocessing import PostProcessor
from orkans.preprocessing import PreProcessor


@logger.catch
def run(
    model_name: str,
    cfg: dict,
    test: bool = False,
):

    tstart = time.perf_counter()

    out_data = {}

    logger.info("Run started.")

    # Number of timesteps to use for velocity field estimation
    n_vsteps = utils.determine_velocity_step_count(model_name, cfg)

    # Load rain rate data
    rainrate_no_transform, metadata_no_transform = utils.load_rainrate_data(
        cfg, n_vsteps
    )

    # PRE-PROCESSING

    pre_processor = PreProcessor(rainrate_no_transform, metadata_no_transform)

    prepro_data = pre_processor.collect_info()

    model_kwargs = utils.load_model_kwargs_from_config(
        model_name, cfg, metadata_no_transform
    )

    run_id = utils.generate_run_id(prepro_data, model_name, model_kwargs)

    out_data = {"id": run_id}

    # Perform nowcast only if more than a certain threshold of cells have precipitation
    ratio = prepro_data["precip_ratio"]
    out_data["precip_ratio"] = ratio

    if ratio < PRECIP_RATIO_THR:
        msg = f"Precipitation ratio below threshold ({ratio} < {PRECIP_RATIO_THR})."
        logger.warning(msg)
        logger.info("Run finished.")
        return out_data

    out_data |= prepro_data

    logger.info(f"Run ID: {run_id}")

    # PROCESSING

    # Automatically transform data using most appropriate transformation
    best_tfunc, Lambda = pre_processor.find_best_transformation()
    rainrate_transform, metadata_transform = pre_processor.apply_transformation(
        best_tfunc,
        Lambda,
    )

    # IMPORTANT!
    # Need to update precipitation threshold since data has changed
    # TODO: Figure out how to implement this in a more fool-proof way
    if "precip_thr" in model_kwargs:
        model_kwargs["precip_thr"] = metadata_transform["threshold"]

    out_data["transform"] = best_tfunc.__name__
    out_data["boxcox_lambda"] = Lambda

    # Check that the grid is equally spaced
    assert metadata_transform["xpixelsize"] == metadata_transform["ypixelsize"]

    # Estimate the motion field

    rainrate_train_transform = rainrate_transform[:n_vsteps, :, :]
    rainrate_train_no_transform = rainrate_no_transform[:n_vsteps, :, :]
    rainrate_valid = rainrate_no_transform[n_vsteps:, :, :]

    # Rain rate with an adjusted (transformed) distribution used for estimating velocity field
    vfield = dense_lucaskanade(rainrate_train_transform)

    # Run nowcast
    n_leadtimes = cfg["general"]["n_leadtimes"]

    model_func = nowcasts.get_method(model_name)

    tstart_nwc = time.perf_counter()

    if model_name in ["linda", "anvil"]:
        nwc = model_func(
            rainrate_train_no_transform,
            vfield,
            n_leadtimes,
            **model_kwargs,
        )

        metadata_nwc = metadata_no_transform

    elif model_name == "sseps":
        nwc = model_func(
            rainrate_train_transform,
            metadata_transform,
            vfield,
            n_leadtimes,
            **model_kwargs,
        )
        nwc, metadata_nwc = utils.conversion.to_rainrate(nwc, metadata_transform)

    elif model_name == "steps":
        nwc = model_func(
            rainrate_train_transform,
            vfield,
            n_leadtimes,
            **model_kwargs,
        )
        nwc, metadata_nwc = utils.conversion.to_rainrate(nwc, metadata_transform)

    else:
        raise NotImplementedError(model_name)

    tend_nwc = time.perf_counter()

    out_data |= model_kwargs

    # POST-PROCESSING

    # All nowcasts should be in mm/h
    # If a model uses different units as input, convert before moving on!

    post_proc = PostProcessor(run_id, rainrate_valid, nwc, metadata_nwc)
    scores = post_proc.calc_scores(0.1, cfg, lead_idx=0)
    out_data |= scores

    out_data["nwc_run_time"] = tend_nwc - tstart_nwc

    logger.info(f"Ran '{model_name}' model")
    logger.info(f"Model settings used: {str(model_kwargs)}")
    logger.info("Run finished.")

    tend = time.perf_counter()
    out_data["total_run_time"] = tend - tstart

    if not test:
        post_proc.save_plots()

    out_data["nwc_model"] = model_name

    return out_data


if __name__ == "__main__":

    # steps, sseps, anvil, linda
    model_name = "steps"

    # Load configuration file
    cfg = utils.load_config()

    run(model_name, cfg)

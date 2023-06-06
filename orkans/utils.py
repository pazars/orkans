import hashlib
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from typing import Union

import numpy as np
import yaml
from loguru import logger
from pysteps import io, rcparams
from pysteps.utils import clip_domain, conversion


def load_production_config() -> dict:
    """Load production YAML configuration file.

    Looks for file config-production.yaml in script directory.

    Returns:
        dict: Configuration file loaded as a dictionary object.
    """

    cfg_prod_path = Path("./config-production.yaml").resolve()
    
    with open(cfg_prod_path.as_posix()) as file:
        return yaml.safe_load(file)

def load_config(cfg_path: Path = None) -> dict:
    """Load YAML configuration file.

    By default looks for file config.yaml in script directory.

    Args:
        cfg_path (Path, optional): Configuration file path. Defaults to None.

    Returns:
        dict: Configuration file loaded as a dictionary object.
    """

    # Load organisation specific pysteps configuration file
    # It does not replace .pystepsrc!
    if not cfg_path:
        script_dir = Path(".").resolve()
        cfg_path = script_dir / "config.yaml"

    with open(cfg_path, "r") as file:
        cfg = yaml.safe_load(file)

    logger.info(f"Read config from {cfg_path.as_posix()}")

    if type(cfg["general"]["datetime"]) != list:
        return [cfg]

    date_cfgs = []
    for date_time in cfg["general"]["datetime"]:
        new_cfg = deepcopy(cfg)
        new_cfg["general"]["datetime"] = date_time
        date_cfgs.append(new_cfg)

    return date_cfgs


def load_batch_config(
    model_name: str, cfg_path: Path = None
) -> Union[list[dict], None]:
    """Load and parse config for batch runs.

    Recongizes which parameters need multiple runs.
    Prepares configurations for each run.

    Args:
        model_name (str): pysteps nowcast model name
        cfg_path (Path, optional): Configuration file path. Defaults to None.

    Returns:
        dict: Configuration file loaded as a dictionary object.
    """
    cfgs = []
    raw_cfgs = load_config(cfg_path)

    # Raw configs differ only in datetime,
    # so just pick batch properties from the first one
    batch_cfg = raw_cfgs[0]["batch"][model_name]

    # Return if batch run not defined
    if not batch_cfg:
        return None

    # Let's just pretend the triple for loop is not there
    for raw_cfg in raw_cfgs:
        for key, values in batch_cfg.items():
            for value in values:
                new_cfg = deepcopy(raw_cfg)
                new_cfg["model"][model_name]["manual"][key] = value
                cfgs.append(new_cfg)

    return cfgs


def load_rainrate_data(cfg: dict, n_vsteps: int) -> tuple[np.ndarray, dict]:
    """Wrapper for pysteps data import.

    Args:
        cfg (dict): Configuration file as dictionary
        n_vsteps (int): Number of timesteps for velocity field estimation

    Returns:
        tuple(np.ndarray, dict): Same (data, metadata) output as in pysteps.
    """

    gen_cfg = cfg["general"]

    data_source = gen_cfg["data_source"]
    start_time = str(gen_cfg["datetime"])
    start_fmt = str(gen_cfg["datetime_fmt"])
    date = datetime.strptime(start_time, start_fmt)

    # Load data source config
    root_path = rcparams.data_sources[data_source]["root_path"]
    path_fmt = rcparams.data_sources[data_source]["path_fmt"]
    fn_pattern = rcparams.data_sources[data_source]["fn_pattern"]
    fn_ext = rcparams.data_sources[data_source]["fn_ext"]
    importer_name = rcparams.data_sources[data_source]["importer"]
    importer_kwargs = rcparams.data_sources[data_source]["importer_kwargs"]
    timestep = rcparams.data_sources[data_source]["timestep"]

    # Load forecast reference data
    n_leadtimes = gen_cfg["n_leadtimes"]

    # Find the radar files in the archive
    filenames = io.find_by_date(
        date,
        root_path,
        path_fmt,
        fn_pattern,
        fn_ext,
        timestep,
        num_prev_files=n_vsteps - 1,
        num_next_files=n_leadtimes,
    )

    # Read the data from the archive
    importer = io.get_method(importer_name, "importer")
    series_data = io.read_timeseries(filenames, importer, **importer_kwargs)
    data, quality, metadata = series_data

    # Data should already be in mm/h, but convert just in case it isn't
    rainrate, metadata = conversion.to_rainrate(data, metadata)

    # Fill missing values with no precipitation value
    rainrate[~np.isfinite(rainrate)] = metadata["zerovalue"]

    # Clip domain to a specific region
    # If first entry is None, uses whole domain
    domain_box = gen_cfg["domain_box"]
    if domain_box[0]:
        # List values parsed as strings, so need to convert back to floats
        domain_box = [float(num) for num in domain_box]
        rainrate, metadata = clip_domain(rainrate, metadata, domain_box)

    return (rainrate, metadata)


def load_rainrate_data_product(cfg: dict, n_vsteps: int, start_time: str) -> tuple[np.ndarray, dict]:
    """Wrapper for pysteps data import.

    Args:
        cfg (dict): Configuration file as dictionary
        n_vsteps (int): Number of timesteps for velocity field estimation
        start_time (str): Last radar image datetime.

    Returns:
        tuple(np.ndarray, dict): Same (data, metadata) output as in pysteps.
    """

    gen_cfg = cfg["general"]

    data_source = gen_cfg["data_source"]
    start_fmt = str(gen_cfg["datetime_fmt"])
    date = datetime.strptime(start_time, start_fmt)

    # Load data source config
    root_path = rcparams.data_sources[data_source]["root_path"]
    path_fmt = rcparams.data_sources[data_source]["path_fmt"]
    fn_pattern = rcparams.data_sources[data_source]["fn_pattern"]
    fn_ext = rcparams.data_sources[data_source]["fn_ext"]
    importer_name = rcparams.data_sources[data_source]["importer"]
    importer_kwargs = rcparams.data_sources[data_source]["importer_kwargs"]
    timestep = rcparams.data_sources[data_source]["timestep"]

    # Find the radar files in the archive
    filenames = io.find_by_date(
        date,
        root_path,
        path_fmt,
        fn_pattern,
        fn_ext,
        timestep,
        num_prev_files=n_vsteps - 1,
    )

    # Read the data from the archive
    importer = io.get_method(importer_name, "importer")
    series_data = io.read_timeseries(filenames, importer, **importer_kwargs)
    data, quality, metadata = series_data

    # Data should already be in mm/h, but convert just in case it isn't
    rainrate, metadata = conversion.to_rainrate(data, metadata)

    # Fill missing values with no precipitation value
    rainrate[~np.isfinite(rainrate)] = metadata["zerovalue"]

    # Clip domain to a specific region
    # If first entry is None, uses whole domain
    domain_box = gen_cfg["domain_box"]
    if domain_box[0]:
        # List values parsed as strings, so need to convert back to floats
        domain_box = [float(num) for num in domain_box]
        rainrate, metadata = clip_domain(rainrate, metadata, domain_box)

    return (rainrate, metadata)


def load_model_kwargs_from_config(model_name: str, cfg: dict, mdata: dict) -> dict:
    """Get nowcast model parameters from configuration file.

    Args:
        model_name (str): Nowcast model name
        cfg (dict): Configuration file loaded as a dictionary
        mdata (dict): Metadata of unmodified pysteps data

    Returns:
        dict: Model parameters
    """
    # Get default nowcast model parameters from configuration file
    model_kwargs = cfg["model"][model_name]["manual"]

    # Update parameters available in metadata, if applicable
    if "metadata" in cfg["model"][model_name]:
        for key, mdkey in cfg["model"][model_name]["metadata"].items():
            model_kwargs[key] = mdata[mdkey]

    return model_kwargs


def determine_velocity_step_count(model_name: str, cfg: dict) -> int:
    """Determiny how many timesteps needed for velocity field estimation.

    Args:
        model_name (str): Nowcast model name
        cfg (dict): Configuration file loaded as a dictionary

    Returns:
        int: Number of timesteps.
    """
    if model_name == "linda":
        order = cfg["model"][model_name]["manual"]["ari_order"]
    else:
        order = cfg["model"][model_name]["manual"]["ar_order"]

    if model_name in ["linda", "anvil"]:
        return order + 2
    return order + 1  # steps, sseps


def generate_run_id(prepro_data: dict, model_name: str, model_kwargs: dict) -> str:
    """Generate a run ID based on info about the data and the nowcast model.

    Args:
        prepro_data (dict): Information about the data from the pre-processor
        model_name (str): Nowcast model name
        model_kwargs (dict): Nowcast model parameters

    Returns:
        str: Run ID
    """
    id_params = {
        "data_date": prepro_data["data_date"],
        "model": model_name,
    }

    id_params |= model_kwargs

    # Generate run ID based on input data and model parameters
    m = hashlib.md5()
    m.update(str(id_params).encode())
    run_id = m.hexdigest()[0:12]
    return run_id

from datetime import datetime
from pathlib import Path
from loguru import logger

import pandas as pd
import numpy as np

from pysteps import rcparams
from pysteps import io
from pysteps.utils import conversion, clip_domain

if __name__ != "__main__":
    from orkans import EVENT_DIR, REGIONS

    EVENT_FILE_PATH = EVENT_DIR / "events.csv"


def _find_last_processed_event(dir: Path, fmt: str = "%Y%m%d%H%M%S") -> datetime:

    logger.info("Checking date of last processed precipitation event")

    last_event_file = dir / "last_processed_event.txt"
    last_event_file_full = last_event_file.resolve().as_posix()

    if not last_event_file.exists():
        logger.info("No previously processed events found")
        return datetime.min

    with open(last_event_file_full, "r") as file:
        last_event_str = file.readlines()[0]
        last_event = datetime.strptime(last_event_str, fmt)

    log_date = last_event.strftime("%Y/%m/%d %H:%M")
    logger.info(f"Date of last processed event: {log_date}")

    return last_event


def _write_latest_event(dir: Path, event: datetime, fmt: str = "%Y%m%d%H%M%S"):

    last_event_file = dir / "last_processed_event.txt"
    last_event_file_full = last_event_file.resolve().as_posix()

    with open(last_event_file_full, "w") as file:
        last_event_str = event.strftime(fmt)
        file.write(last_event_str)


def _load_past_event_data(last_event: datetime):
    if last_event != datetime.min:
        # Load previously processed event file
        return pd.read_csv(EVENT_FILE_PATH)
    return pd.DataFrame()


@logger.catch
def find_new_events(
    data_dir: Path,
    new_data_src: str,
    fmt="%Y%m%d%H%M%S",
):

    logger.info("Looking for new precipitation events")

    last_event = _find_last_processed_event(data_dir)
    data = _load_past_event_data(last_event)

    src_info = rcparams["data_sources"][new_data_src]

    root_path = src_info["root_path"]
    fn_pattern = src_info["fn_pattern"]
    pattern = fn_pattern.strip(fmt) + "*"

    # Recursively search for files names matching pattern
    file_paths = Path(root_path).rglob(pattern)

    new_events = 0
    newest_event = last_event

    dates = np.array(
        [datetime.strptime(fpath.stem.split("_")[-1], fmt) for fpath in file_paths]
    )
    dates = dates[dates > last_event]

    ndates = dates.size

    for count, event in enumerate(dates):
        new_data = process_event(event, new_data_src)
        data = pd.concat([data, new_data])
        new_events += 1
        newest_event = event
        print(f"{count + 1}/{ndates}")

    log_msg = f"Processed {new_events} new events"
    if last_event != datetime.min:
        log_date_from = last_event.strftime("%Y-%m-%d %H:%M")
        log_date_until = newest_event.strftime("%Y-%m-%d %H:%M")
        log_msg += f" from {log_date_from} until {log_date_until}"

    data.to_csv(EVENT_FILE_PATH, index=False)
    _write_latest_event(data_dir, newest_event)

    # logger.info(log_msg)

    return new_events


def process_event(date: datetime, src: str):

    data_source = rcparams.data_sources[src]

    root_path = data_source["root_path"]
    path_fmt = data_source["path_fmt"]
    fn_pattern = data_source["fn_pattern"]
    fn_ext = data_source["fn_ext"]
    importer_name = data_source["importer"]
    importer_kwargs = data_source["importer_kwargs"]
    timestep = data_source["timestep"]

    # Find the input files from the archive
    fns = io.archive.find_by_date(
        date, root_path, path_fmt, fn_pattern, fn_ext, timestep
    )

    # Read the radar composites
    importer = io.get_method(importer_name, "importer")
    # read data, quality rasters, metadata
    R, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)

    # Convert reflectivity to rain rate
    rainrate, metadata = conversion.to_rainrate(R, metadata)

    results = {
        "datetime": [],
        "region_id": [],
        "max_rrate": [],
    }

    date_str = date.strftime("%Y-%m-%d %H:%M")

    for region in REGIONS:
        # rrate_copy = deepcopy(rainrate)
        rrate_region, _ = clip_domain(rainrate, metadata, extent=region["coords"])

        rr_2d = rrate_region[0, :, :]
        # Because there can be a few very large values that don't make sense
        # take 99.9th percentile value as max rainrate.
        max_rate = np.nanpercentile(rr_2d[rr_2d > 0], 99.9)

        results["datetime"].append(date_str)
        results["region_id"].append(region["id"])
        results["max_rrate"].append(max_rate)

    return pd.DataFrame.from_dict(results)


if __name__ == "__main__":
    import sys
    import time

    sys.path.append(Path(".").resolve().as_posix())

    from orkans import EVENT_DIR, REGIONS

    EVENT_FILE_PATH = EVENT_DIR / "events.csv"

    tstart = time.perf_counter()
    find_new_events(EVENT_DIR, "opera_lvgmc")
    tend = time.perf_counter()

    logger.info(f"Runtime: {tend - tstart}s")

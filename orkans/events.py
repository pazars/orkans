from datetime import datetime
from pathlib import Path
from loguru import logger

from pysteps import rcparams


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


@logger.catch
def find_new_events(
    data_dir: Path,
    new_data_src: str,
    fmt="%Y%m%d%H%M%S",
):

    logger.info("Looking for new precipitation events")

    last_event = _find_last_processed_event(data_dir)

    src_info = rcparams["data_sources"][new_data_src]

    root_path = src_info["root_path"]
    fn_pattern = src_info["fn_pattern"]
    pattern = fn_pattern.strip(fmt) + "*"

    # Recursively search for files names matching pattern
    file_paths = Path(root_path).rglob(pattern)

    new_events = []

    for fpath in file_paths:
        event_str = fpath.stem.split("_")[-1]
        event = datetime.strptime(event_str, fmt)
        if event > last_event:
            new_events.append(event)

    log_msg = f"Found {len(new_events)} new events"
    if last_event != datetime.min:
        log_date = last_event.strftime("%Y-%m-%d %H:%M")
        log_msg += f" since {log_date}"

    logger.info(log_msg)

    return new_events

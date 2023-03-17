import argparse
import itertools
import time

from multiprocessing import Pool

from loguru import logger
from orkans import LOG_PATH, CFG_PATH
from orkans import nowcast, utils


if __name__ == "__main__":

    logger.add(
        LOG_PATH,
        backtrace=True,
        diagnose=True,
        rotation="1 day",
        retention="1 week",
    )

    parser = argparse.ArgumentParser(description="Description goes here.")

    parser.add_argument(
        "-m",
        "--models",
        required=True,
        nargs="+",
        help="Nowcast model name(s) (steps, sseps, anvil, linda)",
    )

    parser.add_argument(
        "-n",
        "--nproc",
        type=int,
        default="1",
        help="Number or parallel processes used to run models",
    )

    args = parser.parse_args()

    cfgs = utils.load_and_parse_config(CFG_PATH)

    nwc_args = itertools.product(args.models, cfgs)

    tstart = time.perf_counter()
    with Pool(processes=args.nproc) as pool:
        results = pool.starmap(nowcast.run, nwc_args)
    tend = time.perf_counter()

    print(f"Runtime: {tend - tstart}")

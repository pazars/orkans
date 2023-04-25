import argparse
import itertools
import time
import os
import pandas as pd

from multiprocessing import Pool

from loguru import logger
from orkans import LOG_PATH, CFG_PATH, OUT_DIR
from orkans import nowcast, utils


if __name__ == "__main__":

    # TODO Fix logger for multiprocessing
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

    parser.add_argument(
        "--batch",
        action=argparse.BooleanOptionalAction,
        help="Run models in batch or single run mode",
    )

    args = parser.parse_args()

    user_model_names = args.models

    if not user_model_names:
        logger.info("Nothing to solve. Exiting.")
        exit()

    if args.batch:
        nwc_args = []
        for idx, model_name in enumerate(user_model_names):
            cfgs = utils.load_batch_config(model_name, CFG_PATH)

            if not cfgs:
                # Skip if no batch config defined for model
                user_model_names.pop(idx)
                continue

            # Otherwise prepare input for a nowcast
            nwc_args += list(itertools.product([model_name], cfgs))
    else:
        cfg = utils.load_config()
        nwc_args = itertools.product(user_model_names, cfg)

    tstart = time.perf_counter()
    with Pool(processes=args.nproc) as pool:
        results = pool.starmap(nowcast.run, nwc_args)

    # Save results after all runs are finished
    # This avoids multiple processes trying to access same files

    # Create results directory if it doesn't exist
    if not OUT_DIR.exists():
        os.mkdir(OUT_DIR)

    # Load relevant result files
    model_data_map = {}
    for model_name in user_model_names:

        fname = f"nowcasts_{model_name}.csv"
        out_path = OUT_DIR / fname

        if out_path.exists():
            data = pd.read_csv(out_path)
        else:
            data = pd.DataFrame()

        model_data_map[model_name] = data

    for result in results:

        if not result or "nwc_model" not in result:
            continue

        model_name = result.pop("nwc_model")
        data = model_data_map[model_name]

        new_data = pd.DataFrame.from_dict([result])

        # Output run results
        data = pd.concat([data, new_data])

        model_data_map[model_name] = data

    # Index set to True leads to a redundant column at next read
    for model_name, data in model_data_map.items():
        fname = f"nowcasts_{model_name}.csv"
        out_path = OUT_DIR / fname
        data.to_csv(out_path, index=False)

    tend = time.perf_counter()
    print(f"Runtime: {tend - tstart}")

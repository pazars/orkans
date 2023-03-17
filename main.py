import argparse
import asyncio
import time

from loguru import logger


from orkans import LOG_PATH, CFG_PATH
from orkans import nowcast, utils


async def worker(name, queue):
    while True:
        # Get a "work item" out of the queue.
        sleep_for = await queue.get()

        # Sleep for the "sleep_for" seconds.
        await asyncio.sleep(sleep_for)

        # Notify the queue that the "work item" has been processed.
        queue.task_done()

        print(f"{name} has slept for {sleep_for:.2f} seconds")


async def main():

    queue = asyncio.Queue()

    # Generate random timings and put them into the queue.
    total_sleep_time = 0
    for sleep_for in range(5):
        queue.put_nowait(sleep_for)
        total_sleep_time += sleep_for

    # Create three worker tasks to process the queue concurrently.
    tasks = []
    for i in range(3):
        task = asyncio.create_task(worker(f"worker-{i}", queue))
        tasks.append(task)

    # Wait until the queue is fully processed.
    started_at = time.monotonic()
    await queue.join()
    total_slept_for = time.monotonic() - started_at

    # Cancel our worker tasks.
    for task in tasks:
        task.cancel()
    # Wait until all worker tasks are cancelled.
    await asyncio.gather(*tasks, return_exceptions=True)

    print("====")
    print(f"3 workers slept in parallel for {total_slept_for:.2f} seconds")
    print(f"total expected sleep time: {total_sleep_time} seconds")


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
        "-w",
        "--workers",
        nargs=1,
        type=int,
        default=1,
        help="Number or parallel workers used to run models",
    )

    args = parser.parse_args()
    print(args.models)

    cfgs = utils.load_and_parse_config(CFG_PATH)

    nwc_arg_zip = zip(args.models, cfgs)
    for nwc_args in nwc_arg_zip:
        model_name, cfg = nwc_args
        nowcast.run(model_name, cfg)

    # asyncio.run(main())

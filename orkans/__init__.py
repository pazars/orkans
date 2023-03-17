# Log file path for loguru
from pathlib import Path

# Configuration file path
CFG_PATH = (Path(".") / "config.yaml").resolve()

# Log file path
LOG_PATH = (Path(".") / "logs" / "log.txt").resolve()

# Result path
OUT_DIR = (Path(".") / "results").resolve()

# Result plot directory
PLOT_DIR = OUT_DIR / "plots"

# Precipitation ratio threshold
PRECIP_RATIO_THR = 0.25

# Log file path for loguru
from pathlib import Path

ROOT_DIR = Path(".")

# Configuration file path
CFG_PATH = (ROOT_DIR / "config.yaml").resolve()

# Log file path
LOG_PATH = (ROOT_DIR / "logs" / "log.txt").resolve()

# Result path
OUT_DIR = (ROOT_DIR / "results").resolve()

# Result plot directory
PLOT_DIR = OUT_DIR / "plots"

# Precipitation ratio threshold
PRECIP_RATIO_THR = 0.25

# Test config directory
TEST_CFG_DIR = ROOT_DIR / "orkans" / "tests" / "_test_configs"

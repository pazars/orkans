# Log file path for loguru
from pathlib import Path

ROOT_DIR = Path(".")

# Configuration file path
CFG_PATH = (ROOT_DIR / "config.yaml").resolve()

# Log file path
LOG_PATH = (ROOT_DIR / "logs" / "log.txt").resolve()

# Result path
OUT_DIR = (ROOT_DIR / "results").resolve()

# Result path
EVENT_DIR = (ROOT_DIR / "events").resolve()

# Result plot directory
PLOT_DIR = OUT_DIR / "plots"

# Precipitation ratio threshold
PRECIP_RATIO_THR = 0.15

# Test config directory
TEST_CFG_DIR = ROOT_DIR / "orkans" / "tests" / "_test_configs"

REGIONS = [
    {
        "id": 1,
        "description": "Estonia",
        "coords": (2.6e6, 3.009e6, -1.719e6, -1.481e6),
    },
    {
        "id": 2,
        "description": "Sweden's islands in the Baltic sea",
        "coords": (2.284e6, 2.622e6, -1.955e6, -1.664e6),
    },
]

"""Repository-relative locations, resolved from this file and never from the
working directory. The package is installed editable (pip install -e .), so
this file lives inside the checkout and REPO is the checkout. Nothing here
creates a directory.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_REAL = REPO / "data" / "real"      # the _01 (failure) and _02 (good) captures
DATA_SIM = REPO / "data" / "sim"        # committed sim_*.txt fixtures; generated CSVs
CAL_DATA = REPO / "cal_data"            # bench calibration captures + manifest
MODELS = REPO / "models"                # gain_match.json, model_*.json
FIGURES = REPO / "figures"
DOCS = REPO / "docs"

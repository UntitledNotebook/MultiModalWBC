"""Asset directory configuration for robot models."""

import pathlib

# Path to the unitree_model directory containing robot USD files
ASSET_DIR = pathlib.Path(__file__).parents[3] / "third_party"

"""
Configuration for Sionna IEEE experiments.
"""
import os
from pathlib import Path
import numpy as np


# Project paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
SCENES_DIR = PROJECT_DIR / "scenes"
MESHES_DIR = PROJECT_DIR / "meshes"

# Ensure directories exist
for dir_path in [DATA_DIR, RESULTS_DIR, SCENES_DIR, MESHES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Rome center coordinates (for coordinate conversion)
ROME_CENTER = {
    "lat": 41.8698541,
    "lon": 12.4622493,
}

# Scene bounds for OSM data fetching (from actual experiments)
SCENE_BOUNDS = {
    "lat_range": (41.861824, 41.8778842),
    "lon_range": (12.4525696, 12.471929),
}

# Default experiment configuration
DEFAULT_CONFIG = {
    "data_path": DATA_DIR / "rome_crop_knn_ready.csv",
    "scene_path": SCENES_DIR / "rome_scene_with_heights.xml",
    "output_dir": RESULTS_DIR,
    "center_lat": ROME_CENTER["lat"],
    "center_lon": ROME_CENTER["lon"],
}

# Experiment parameters
DEFAULT_EXP_PARAMS = {
    "name": "default",
    "tx_height": 60,
    "max_depth": 3,
    "max_num_paths_per_src": 10000,
    "samples_per_src": int(1e6),
    "synthetic_array": False,
    "los": True,
    "specular_reflection": False,
    "diffuse_reflection": True,
    "refraction": True,
    "seed": 40,
    "num_rows": 6,
    "num_cols": 6,
    "batch_size": 512,
    "tx_pattern": "hw_dipole",
    "tx_polarization": "VH",
    "rx_pattern": "hw_dipole",
    "rx_polarization": "cross",
    "frequency": int(1.2e9),
    "tx_orientation": [0, 0, 0],
    "rx_orientation": [0, 0, 0],
}

# Best per-BS parameters from experiments
BEST_PER_BS_PARAMS = {
    0: {"tx_orientation": [300, 0, 0], "tx_height": 60},
    1: {"tx_orientation": [0, -5, 300]},
    2: {"tx_height": 55},
    3: {"tx_orientation": [90, -5, 0]},
    4: {"tx_orientation": [90, -5, 0]},
    5: {"tx_orientation": [150, -5, 0]},
}

# Parameter sweep configurations
PARAMETER_SWEEPS = {
    "tx_height": {
        "values": [11, 12, 15, 20, 40, 55],
        "base_params": DEFAULT_EXP_PARAMS.copy(),
    },
    "frequency": {
        "values": [1.01e9, 1.05e9, 1.1e9, 1.2e9, 1.3e9, 1.5e9, 2e9, 3.6e9, 5e9],
        "base_params": DEFAULT_EXP_PARAMS.copy(),
    },
    "tx_pattern": {
        "values": ["iso", "dipole", "hw_dipole", "tr38901"],
        "base_params": DEFAULT_EXP_PARAMS.copy(),
    },
    "rx_pattern": {
        "values": ["iso", "dipole", "hw_dipole", "tr38901"],
        "base_params": DEFAULT_EXP_PARAMS.copy(),
    },
    "max_depth": {
        "values": [1, 2, 3, 4, 5, 6, 7],
        "base_params": DEFAULT_EXP_PARAMS.copy(),
    },
}

# Azimuth sweep configuration
def get_azimuth_sweep_config(pattern_type="tx"):
    """Get azimuth sweep configuration for TX or RX."""
    base_params = DEFAULT_EXP_PARAMS.copy()
    if pattern_type == "tx":
        base_params["tx_pattern"] = "tr38901"
        param_name = "tx_orientation"
    else:
        base_params["rx_pattern"] = "hw_dipole"
        param_name = "rx_orientation"
    
    values = []
    for azimuth in range(0, 360, 30):
        values.append([azimuth, -5, 0])
    
    return {
        "param_name": param_name,
        "values": values,
        "base_params": base_params,
    }
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from sionna.rt import load_scene

from .simulation import run_simulation, evaluate_simulation
from .data_processing import (
    load_rome_data, correct_bs_coordinates, create_tx_rx_maps, 
    create_data_arrays, BS_CORRECTIONS
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        self._prepare_data()
        
        # Load scene
        self.scene = load_scene(config['scene_path'])
        logger.info(f"Loaded scene: {config['scene_path']}")
    
    def _prepare_data(self):
        # Load data
        df = load_rome_data(self.config['data_path'])
        
        # Apply BS corrections
        df_corrected = correct_bs_coordinates(df, BS_CORRECTIONS)
        
        # Create mappings
        self.tx_map, self.rx_map = create_tx_rx_maps(df_corrected)
        
        # Create data arrays
        self.rome_data = create_data_arrays(df_corrected, self.tx_map, self.rx_map)
        
        logger.info(f"Prepared data: {len(self.tx_map)} BSs, {len(self.rx_map)} UEs")
    
    def run_single_experiment(self, exp_params: Dict, save_results: bool = True) -> List[float]:
        # Prepare filenames
        exp_name = exp_params['name']
        filename = self.output_dir / f"{exp_name}.npy"
        config_filename = self.output_dir / f"{exp_name}_config.npy"
        
        # Check if already exists
        if filename.exists() and save_results:
            logger.warning(f"Results already exist: {filename}")
            return None
        
        # Save config
        if save_results:
            np.save(config_filename, exp_params)
        
        # Run simulation
        metrics = run_simulation(
            self.scene,
            self.tx_map,
            self.rx_map,
            exp_params,
            self.config['center_lon'],
            self.config['center_lat']
        )
        
        if metrics is None:
            logger.error("Simulation failed")
            return None
        
        rssi_dbm, nsinr_db, nrsrp_dbm, nrsrq_db = metrics
        
        # result array
        n_bs = len(self.tx_map)
        n_ue = len(self.rx_map)
        sionna_data = np.zeros((n_bs, n_ue, 6))
        
        # Fill metrics - transposed to match expected shape
        sionna_data[:, :, 0] = rssi_dbm.T
        sionna_data[:, :, 1] = nsinr_db.T
        sionna_data[:, :, 2] = nrsrp_dbm.T
        sionna_data[:, :, 3] = nrsrq_db.T
        # Keep lat/lon as zeros (not used in simulation)
        
        # Evaluate
        spearmans = evaluate_simulation(sionna_data, self.rome_data, self.tx_map)
        
        # Save results
        if save_results:
            np.save(filename, sionna_data)
            logger.info(f"Saved results to {filename}")
        
        return spearmans
    
    def run_parameter_sweep(self, param_name: str, param_values: List, 
                           base_params: Dict, plot_results: bool = True) -> Dict:
        results = {}
        
        for value in param_values:
            # Update parameters
            exp_params = base_params.copy()
            exp_params[param_name] = value
            exp_params['name'] = f"{param_name}_{value}"
            
            # Run experiment
            spearmans = self.run_single_experiment(exp_params)
            
            if spearmans is not None:
                results[value] = spearmans
        
        # Plot results
        if plot_results and results:
            self._plot_parameter_sweep(param_name, results)
        
        return results
    
    def run_per_bs_experiment(self, base_params: Dict, per_bs_params: Dict) -> List[float]:
        #  filenames
        exp_name = base_params['name']
        filename = self.output_dir / f"{exp_name}.npy"
        config_filename = self.output_dir / f"{exp_name}_config.npy"
        
        # Check if already exists
        if filename.exists():
            logger.warning(f"Results already exist: {filename}")
            return None
        
        # Save config (both base and per-BS params)
        run_config = {
            "global_params": base_params.copy(),
            "per_bs_params": per_bs_params.copy(),
        }
        np.save(config_filename, run_config)
        
        # Run simulation with per-BS overrides
        metrics = run_simulation(
            self.scene,
            self.tx_map,
            self.rx_map,
            base_params,
            self.config['center_lon'],
            self.config['center_lat'],
            exp_params_per_bs=per_bs_params
        )
        
        if metrics is None:
            logger.error("Simulation failed")
            return None
        
        rssi_dbm, nsinr_db, nrsrp_dbm, nrsrq_db = metrics
        
        #  result array
        n_bs = len(self.tx_map)
        n_ue = len(self.rx_map)
        sionna_data = np.zeros((n_bs, n_ue, 6))
        
        # Fill in metrics
        sionna_data[:, :, 0] = rssi_dbm.T
        sionna_data[:, :, 1] = nsinr_db.T
        sionna_data[:, :, 2] = nrsrp_dbm.T
        sionna_data[:, :, 3] = nrsrq_db.T
        
        # Evaluate
        spearmans = evaluate_simulation(sionna_data, self.rome_data, self.tx_map)
        
        # Save results
        np.save(filename, sionna_data)
        logger.info(f"Saved results to {filename}")
        
        return spearmans
    
    def _plot_parameter_sweep(self, param_name: str, results: Dict):
        plt.figure(figsize=(8, 5))
        
        for key, spearmans in results.items():
            if spearmans is not None:
                plt.plot(spearmans, label=str(key), marker='o')
        
        plt.xlabel("Base Station Index")
        plt.ylabel("Spearman Correlation")
        plt.title(f"Parameter Sweep: {param_name}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"sweep_{param_name}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved plot to {plot_path}")


# Default experiment parameters
DEFAULT_EXP_PARAMS = {
    "name": "default",
    "tx_height": 40,
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
    "tx_pattern": "tr38901",
    "tx_polarization": "VH",
    "rx_pattern": "hw_dipole",
    "rx_polarization": "cross",
    "frequency": int(1.2e9),
    "tx_orientation": [0, 0, 0],
    "rx_orientation": [0, 0, 0],
}

# Best per-BS parameters from paper
BEST_PER_BS_PARAMS = {
    0: {"rx_orientation": [120, -5, 0], "tx_height": 40},
    1: {"tx_orientation": [240, -5, 0]},
    2: {"tx_height": 55},
    3: {"tx_orientation": [90, -5, 0]},
    4: {"tx_orientation": [90, -5, 0]},
    5: {"rx_orientation": [150, -5, 0]},
}
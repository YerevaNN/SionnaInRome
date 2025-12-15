import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.experiment_runner import ExperimentRunner
from config.config import (
    DEFAULT_CONFIG, DEFAULT_EXP_PARAMS, BEST_PER_BS_PARAMS,
    PARAMETER_SWEEPS, get_azimuth_sweep_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_baseline_experiment(runner: ExperimentRunner, scene_suffix: str = ""):
    """Run baseline experiment with default parameters."""
    logger.info("Running baseline experiment...")
    
    exp_params = DEFAULT_EXP_PARAMS.copy()
    exp_params['name'] = f'baseline_default{scene_suffix}'
    
    spearmans = runner.run_single_experiment(exp_params)
    
    if spearmans:
        logger.info("Baseline results:")
        for i, corr in enumerate(spearmans):
            logger.info(f"  BS {i}: {corr:.3f}")
    
    return spearmans


def run_optimized_experiment(runner: ExperimentRunner, scene_suffix: str = ""):
    """Run experiment with optimized per-BS parameters."""
    logger.info("Running optimized per-BS experiment...")
    
    exp_params = DEFAULT_EXP_PARAMS.copy()
    exp_params['name'] = f'optimized_per_bs{scene_suffix}'
    
    spearmans = runner.run_per_bs_experiment(exp_params, BEST_PER_BS_PARAMS)
    
    if spearmans:
        logger.info("Optimized results:")
        for i, corr in enumerate(spearmans):
            logger.info(f"  BS {i}: {corr:.3f}")
    
    return spearmans


def run_parameter_sweeps(runner: ExperimentRunner):
    """Run all parameter sweep experiments."""
    logger.info("Running parameter sweeps...")
    
    results = {}
    
    # Standard parameter sweeps
    for param_name, config in PARAMETER_SWEEPS.items():
        logger.info(f"\nSweeping parameter: {param_name}")
        sweep_results = runner.run_parameter_sweep(
            param_name,
            config['values'],
            config['base_params']
        )
        results[param_name] = sweep_results
    
    # TX azimuth sweep
    logger.info("\nSweeping TX azimuth...")
    tx_azimuth_config = get_azimuth_sweep_config("tx")
    tx_azimuth_results = runner.run_parameter_sweep(
        tx_azimuth_config['param_name'],
        tx_azimuth_config['values'],
        tx_azimuth_config['base_params']
    )
    results['tx_azimuth'] = tx_azimuth_results
    
    # RX azimuth sweep
    logger.info("\nSweeping RX azimuth...")
    rx_azimuth_config = get_azimuth_sweep_config("rx")
    rx_azimuth_results = runner.run_parameter_sweep(
        rx_azimuth_config['param_name'],
        rx_azimuth_config['values'],
        rx_azimuth_config['base_params']
    )
    results['rx_azimuth'] = rx_azimuth_results
    
    return results


def get_scene_suffix(scene_path: Path) -> str:
    """
    Extract suffix from scene filename to use in experiment names.
    
    Examples:
        'rome_scene_with_heights.xml' -> ''
        'rome_scene_tapered_buildings.xml' -> '_tapered_buildings'
        'rome_scene_taller_buildings.xml' -> '_taller_buildings'
    """
    scene_name = scene_path.stem  # Get filename without extension
    
    # Remove 'rome_scene' prefix
    if scene_name.startswith('rome_scene'):
        suffix = scene_name[len('rome_scene'):]
        # Keep the suffix if it's not the default '_with_heights'
        if suffix and suffix != '_with_heights':
            return suffix
    
    return ""


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Sionna IEEE experiments")
    parser.add_argument(
        '--experiment', 
        choices=['baseline', 'optimized', 'sweeps', 'all'],
        default='all',
        help='Which experiment(s) to run'
    )
    parser.add_argument(
        '--data-path',
        help='Path to Rome CSV data file'
    )
    parser.add_argument(
        '--scene-path',
        help='Path to Sionna scene XML file'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory for output results'
    )
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    if args.data_path:
        config['data_path'] = Path(args.data_path)
    if args.scene_path:
        config['scene_path'] = Path(args.scene_path)
    if args.output_dir:
        config['output_dir'] = Path(args.output_dir)
    
    # Check if required files exist
    if not config['data_path'].exists():
        logger.error(f"Data file not found: {config['data_path']}")
        logger.info("Please copy rome_crop_knn_ready.csv to sionna_ieee/data/")
        return
    
    if not config['scene_path'].exists():
        logger.error(f"Scene file not found: {config['scene_path']}")
        logger.info("Please run generate_scene.py first to create the scene")
        return
    
    # Extract scene suffix for experiment naming
    scene_suffix = get_scene_suffix(config['scene_path'])
    if scene_suffix:
        logger.info(f"Using scene variant: {scene_suffix}")
    
    # Initialize experiment runner
    try:
        runner = ExperimentRunner(config)
    except Exception as e:
        logger.error(f"Failed to initialize experiment runner: {e}")
        return
    
    # Run experiments
    if args.experiment in ['baseline', 'all']:
        run_baseline_experiment(runner, scene_suffix)
    
    if args.experiment in ['optimized', 'all']:
        run_optimized_experiment(runner, scene_suffix)
    
    if args.experiment in ['sweeps', 'all']:
        run_parameter_sweeps(runner)
    
    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()
"""
Comprehensive per-BS parameter optimization with combination testing.
This will find the best parameter combinations for each base station.
"""
import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from itertools import product
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.experiment_runner import ExperimentRunner
from config.config import DEFAULT_CONFIG, DEFAULT_EXP_PARAMS

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Opt. parameter ranges
# Angles are in degrees here, will be converted to radians
OPTIMIZATION_PARAMS = {
    "tx_height": [15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    
    # α (alpha) - Azimuth angle (horizontal rotation around Z-axis)
    "tx_azimuth": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    
    # β (beta) - Elevation/tilt angle (vertical tilt)
    "tx_elevation": [-30, -15, -10, -5, 0, 5, 10, 15, 30],
    
    # γ (gamma) - Roll angle (rotation around forward axis)
    "tx_roll": [-45, -30, -15, 0, 15, 30, 45],
}

def deg_to_rad(degrees):
    return np.deg2rad(degrees)

def create_bs_config(bs_idx: int, height: float = None, 
                     azimuth: float = None, elevation: float = None, 
                     roll: float = None):
    config = {}
    
    if height is not None:
        config['tx_height'] = height
    
    # Build orientation if any angle is specified
    if any(angle is not None for angle in [azimuth, elevation, roll]):
        orientation = [
            deg_to_rad(azimuth) if azimuth is not None else 0,
            deg_to_rad(elevation) if elevation is not None else 0,
            deg_to_rad(roll) if roll is not None else 0
        ]
        config['tx_orientation'] = orientation
    
    return {bs_idx: config} if config else {}

def optimize_progressive(runner: ExperimentRunner, bs_idx: int, 
                        param_ranges: dict, max_experiments: int = 100,
                        optimization_mode: str = 'progressive'):
    logger.info(f"Optimizing BS {bs_idx} using {optimization_mode} mode")
    
    results = []
    experiment_count = 0
    
    if optimization_mode == 'independent':
        # Original approach - test each parameter independently
        best_config = optimize_independent(runner, bs_idx, param_ranges, max_experiments)
        
    elif optimization_mode == 'progressive':
        # Progressive optimization - build up best configuration step by step
        best_config = optimize_progressive_internal(runner, bs_idx, param_ranges, max_experiments)
        
    elif optimization_mode == 'combinations':
        # Test selected combinations
        best_config = optimize_combinations(runner, bs_idx, param_ranges, max_experiments)
    
    else:
        raise ValueError(f"Unknown optimization mode: {optimization_mode}")
    
    return best_config

def optimize_independent(runner, bs_idx, param_ranges, max_experiments):
    results = []
    experiments_per_param = max_experiments // len(param_ranges)
    
    for param_name, param_values in param_ranges.items():
        logger.info(f"  Testing {param_name} for BS {bs_idx}")
        
        # Sample values if too many
        if len(param_values) > experiments_per_param:
            param_values = np.random.choice(param_values, experiments_per_param, replace=False)
        
        for value in param_values:
            # configuration with only this parameter
            if param_name == "tx_height":
                config = create_bs_config(bs_idx, height=value)
            elif param_name == "tx_azimuth":
                config = create_bs_config(bs_idx, azimuth=value)
            elif param_name == "tx_elevation":
                config = create_bs_config(bs_idx, elevation=value)
            elif param_name == "tx_roll":
                config = create_bs_config(bs_idx, roll=value)
            
            result = run_experiment(runner, bs_idx, config, f"{param_name}_{value}")
            if result:
                result['param_name'] = param_name
                result['param_value'] = value
                results.append(result)
    
    return analyze_results(results, bs_idx)

def optimize_progressive_internal(runner, bs_idx, param_ranges, max_experiments):
    results = []
    best_values = {}
    
    # Order of optimization 
    param_order = ['tx_height', 'tx_azimuth', 'tx_elevation', 'tx_roll']
    
    experiments_per_param = max_experiments // len(param_order)
    
    for param_name in param_order:
        if param_name not in param_ranges:
            continue
            
        logger.info(f"  Progressive optimization of {param_name} for BS {bs_idx}")
        logger.info(f"    Using best values so far: {best_values}")
        
        param_values = param_ranges[param_name]
        
        # Sample 
        if len(param_values) > experiments_per_param:
            param_values = np.random.choice(param_values, experiments_per_param, replace=False)
        
        best_correlation = -1
        best_value = None
        
        for value in param_values:
            # config with all best values found so far + current test value
            config = create_bs_config(
                bs_idx,
                height=best_values.get('tx_height', None) if param_name != 'tx_height' else value,
                azimuth=best_values.get('tx_azimuth', None) if param_name != 'tx_azimuth' else value,
                elevation=best_values.get('tx_elevation', None) if param_name != 'tx_elevation' else value,
                roll=best_values.get('tx_roll', None) if param_name != 'tx_roll' else value
            )
            
            exp_name = f"prog_{param_name}_{value}"
            for k, v in best_values.items():
                exp_name += f"_{k[3:]}={v}"  # context of other params
            
            result = run_experiment(runner, bs_idx, config, exp_name)
            
            if result:
                result['param_name'] = param_name
                result['param_value'] = value
                result['used_values'] = best_values.copy()
                results.append(result)
                
                if result['correlation'] > best_correlation:
                    best_correlation = result['correlation']
                    best_value = value
                    
                logger.info(f"      {param_name}={value}: ρ={result['correlation']:.3f}")
        
        # Update best values with the best value found for this parameter
        if best_value is not None:
            best_values[param_name] = best_value
            logger.info(f"    Best {param_name}: {best_value} (ρ={best_correlation:.3f})")
    
    # Create final configuration with all best values
    final_config = create_bs_config(
        bs_idx,
        height=best_values.get('tx_height'),
        azimuth=best_values.get('tx_azimuth'),
        elevation=best_values.get('tx_elevation'),
        roll=best_values.get('tx_roll')
    )
    
    return {
        'bs_idx': bs_idx,
        'best_values': best_values,
        'final_config': final_config[bs_idx] if bs_idx in final_config else {},
        'all_results': results,
        'optimization_mode': 'progressive'
    }

def optimize_combinations(runner, bs_idx, param_ranges, max_experiments):
    results = []
    
    # Create smart combinations to test
    combinations = []
    
    # 1. Add corners of the parameter space
    for height in [min(param_ranges['tx_height']), max(param_ranges['tx_height'])]:
        for azimuth in [0, 90, 180, 270]:
            for elevation in [min(param_ranges['tx_elevation']), 0, max(param_ranges['tx_elevation'])]:
                combinations.append({
                    'tx_height': height,
                    'tx_azimuth': azimuth,
                    'tx_elevation': elevation,
                    'tx_roll': 0  # Usually keep roll at 0
                })
    
    # 2. Add some random combinations
    n_random = min(max_experiments - len(combinations), 20)
    for _ in range(n_random):
        combinations.append({
            'tx_height': np.random.choice(param_ranges['tx_height']),
            'tx_azimuth': np.random.choice(param_ranges['tx_azimuth']),
            'tx_elevation': np.random.choice(param_ranges['tx_elevation']),
            'tx_roll': np.random.choice([0, 0, 0, 15, -15])  # Mostly 0
        })
    
    # 3. Test combinations
    logger.info(f"  Testing {len(combinations)} combinations for BS {bs_idx}")
    
    best_correlation = -1
    best_combo = None
    
    for i, combo in enumerate(combinations[:max_experiments]):
        config = create_bs_config(
            bs_idx,
            height=combo['tx_height'],
            azimuth=combo['tx_azimuth'],
            elevation=combo['tx_elevation'],
            roll=combo['tx_roll']
        )
        
        exp_name = f"combo_{i}_h{combo['tx_height']}_az{combo['tx_azimuth']}_el{combo['tx_elevation']}"
        result = run_experiment(runner, bs_idx, config, exp_name)
        
        if result:
            result['combination'] = combo
            results.append(result)
            
            if result['correlation'] > best_correlation:
                best_correlation = result['correlation']
                best_combo = combo
            
            if (i + 1) % 10 == 0:
                logger.info(f"    Tested {i+1} combinations, best so far: ρ={best_correlation:.3f}")
    
    # Create final configuration
    if best_combo:
        final_config = create_bs_config(
            bs_idx,
            height=best_combo['tx_height'],
            azimuth=best_combo['tx_azimuth'],
            elevation=best_combo['tx_elevation'],
            roll=best_combo['tx_roll']
        )
        
        logger.info(f"  Best combination: {best_combo} (ρ={best_correlation:.3f})")
        
        return {
            'bs_idx': bs_idx,
            'best_values': best_combo,
            'final_config': final_config[bs_idx] if bs_idx in final_config else {},
            'all_results': results,
            'optimization_mode': 'combinations'
        }
    
    return None

def run_experiment(runner, bs_idx, per_bs_config, exp_name):
    try:
        exp_params = DEFAULT_EXP_PARAMS.copy()
        exp_params['name'] = f'opt_bs{bs_idx}_{exp_name}'
        
        if per_bs_config:
            spearmans = runner.run_per_bs_experiment(exp_params, per_bs_config)
        else:
            spearmans = runner.run_single_experiment(exp_params)
        
        if spearmans:
            return {
                'bs_idx': bs_idx,
                'correlation': spearmans[bs_idx],
                'all_correlations': spearmans,
                'experiment_name': exp_params['name'],
                'config': per_bs_config
            }
    except Exception as e:
        logger.error(f"    Failed experiment {exp_name}: {e}")
    
    return None

def analyze_results(results, bs_idx):
    if not results:
        return None
    
    # Find best result
    best_result = max(results, key=lambda x: x['correlation'])
    
    # Extract configuration from best result
    final_config = best_result.get('config', {}).get(bs_idx, {})
    
    return {
        'bs_idx': bs_idx,
        'best_result': best_result,
        'final_config': final_config,
        'all_results': results
    }

def optimize_all_bs(optimization_mode='progressive', max_experiments_per_bs=60):
    logger.info(f"Starting per-BS optimization in {optimization_mode} mode")
    logger.info(f"Max experiments per BS: {max_experiments_per_bs}")
    
    # Initialize experiment runner
    runner = ExperimentRunner(DEFAULT_CONFIG)
    
    # Results storage
    all_results = {}
    best_per_bs_config = {}
    
    # Create results directory
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Optimize each BS
    for bs_idx in range(6):  # 6 base stations
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing BS {bs_idx}")
        logger.info(f"{'='*60}")
        
        bs_results = optimize_progressive(
            runner, bs_idx, OPTIMIZATION_PARAMS, 
            max_experiments_per_bs, optimization_mode
        )
        
        if bs_results:
            all_results[bs_idx] = bs_results
            
            # Save individual BS results
            bs_file = results_dir / f"bs_{bs_idx}_{optimization_mode}_{timestamp}.json"
            with open(bs_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                bs_results_json = json.loads(json.dumps(bs_results, default=str))
                json.dump(bs_results_json, f, indent=2)
            
            # Extract configuration
            if 'final_config' in bs_results and bs_results['final_config']:
                best_per_bs_config[bs_idx] = bs_results['final_config']
    
    # Generate summary report
    print("\n" + "="*80)
    print(f"OPTIMIZATION SUMMARY ({optimization_mode} mode)")
    print("="*80)
    
    for bs_idx, results in all_results.items():
        print(f"\nBS {bs_idx}:")
        if 'best_values' in results:
            print(f"  Best values: {results['best_values']}")
        if 'final_config' in results:
            config = results['final_config']
            if 'tx_orientation' in config:
                ori_rad = config['tx_orientation']
                ori_deg = [np.rad2deg(angle) for angle in ori_rad]
                print(f"  Height: {config.get('tx_height', 'default')}m")
                print(f"  Orientation (deg): azimuth={ori_deg[0]:.1f}°, elevation={ori_deg[1]:.1f}°, roll={ori_deg[2]:.1f}°")
    
    print("\n" + "="*80)
    print("Generated optimal per-BS configuration:")
    print("BEST_PER_BS_PARAMS = {")
    for bs_idx, config in best_per_bs_config.items():
        if 'tx_orientation' in config:
            ori_rad = config['tx_orientation']
            ori_deg = [np.rad2deg(angle) for angle in ori_rad]
            print(f"    {bs_idx}: {{")
            print(f"        'tx_height': {config.get('tx_height', 30)},")
            print(f"        'tx_orientation': [{ori_rad[0]:.4f}, {ori_rad[1]:.4f}, {ori_rad[2]:.4f}],  # deg: [{ori_deg[0]:.0f}, {ori_deg[1]:.0f}, {ori_deg[2]:.0f}]")
            print(f"    }},")
        else:
            print(f"    {bs_idx}: {config},")
    print("}")
    
    # Save complete results
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'optimization_mode': optimization_mode,
        'best_per_bs_config': best_per_bs_config,
        'all_results': {str(k): v for k, v in all_results.items()},
        'parameter_ranges': OPTIMIZATION_PARAMS
    }
    
    summary_file = results_dir / f"optimization_summary_{optimization_mode}_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    logger.info(f"\nOptimization complete! Results saved in {results_dir}")
    
    return best_per_bs_config

def quick_optimization(mode='progressive'):
    logger.info(f"Running quick optimization in {mode} mode")
    
    quick_params = {
        "tx_height": [25, 35, 45, 55],
        "tx_azimuth": [0, 60, 120, 180, 240, 300],
        "tx_elevation": [-15, -5, 0, 5, 15],
        "tx_roll": [0],  # Usually keep at 0 for quick tests
    }
    
    # Replace global params
    global OPTIMIZATION_PARAMS
    original_params = OPTIMIZATION_PARAMS.copy()
    OPTIMIZATION_PARAMS = quick_params
    
    try:
        result = optimize_all_bs(optimization_mode=mode, max_experiments_per_bs=30)
        return result
    finally:
        # Restore original params
        OPTIMIZATION_PARAMS = original_params

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize per-BS parameters")
    parser.add_argument('--mode', type=str, 
                       choices=['independent', 'progressive', 'combinations'],
                       default='progressive',
                       help='Optimization mode: independent (fast), progressive (balanced), combinations (thorough)')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick optimization with reduced parameter ranges')
    parser.add_argument('--max-experiments', type=int, default=60,
                       help='Maximum experiments per BS')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_optimization(mode=args.mode)
    else:
        optimize_all_bs(optimization_mode=args.mode, 
                       max_experiments_per_bs=args.max_experiments)
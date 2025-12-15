import os
import sys
import logging
import argparse
import glob
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.knn_evaluation import knn_eval, analyze_rssi_experiments
from config.config import RESULTS_DIR, DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_single_result(sim_path: str, gt_path: str):
    """Evaluate a single simulation result."""
    logger.info(f"Evaluating: {os.path.basename(sim_path)}")
    
    # Evaluate with RSSI (feature_index=0)
    results_rssi = knn_eval(sim_path, gt_path, feature_index=0)
    logger.info("RSSI-based KNN results:")
    print(results_rssi.to_string(index=False))
    print("\n" + "-"*50 + "\n")
    
    return results_rssi


def evaluate_all_results(results_dir: Path, gt_path: str):
    """Evaluate all simulation results in a directory."""
    # Find all .npy files (excluding config files)
    npy_files = [f for f in glob.glob(str(results_dir / "*.npy"))
                 if "config" not in os.path.basename(f)]
    
    if not npy_files:
        logger.warning(f"No result files found in {results_dir}")
        return
    
    logger.info(f"Found {len(npy_files)} result files to evaluate")
    
    all_results = []
    
    for npy_file in sorted(npy_files):
        result = evaluate_single_result(npy_file, gt_path)
        
        # Add experiment name to results
        exp_name = os.path.basename(npy_file).replace('.npy', '')
        result['Experiment'] = exp_name
        all_results.append(result)
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save 
        output_path = results_dir / "knn_evaluation_summary.csv"
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Saved combined results to {output_path}")
        
        print("\nSummary Statistics:")
        print("==================")
        
        # Group by experiment and show mean errors
        summary = combined_df.groupby('Experiment')['Mean Haversine (m)'].agg(['mean', 'std'])
        print(summary)


def analyze_rssi_statistics(results_dir: Path, gt_path: str):
    """Analyze RSSI statistics for all experiments."""
    # Find all .npy files (excluding config files)
    npy_files = [f for f in glob.glob(str(results_dir / "*.npy"))
                 if "config" not in os.path.basename(f)]
    
    if not npy_files:
        logger.warning(f"No result files found in {results_dir}")
        return
    
    logger.info(f"Analyzing RSSI statistics for {len(npy_files)} files...")
    
    # Analyze all experiments
    df_metrics = analyze_rssi_experiments(npy_files, gt_path)
    
    # save
    output_path = results_dir / "rssi_analysis_summary.csv"
    df_metrics.to_csv(output_path, index=False)
    logger.info(f"Saved RSSI analysis to {output_path}")
    
    print("\nRSSI Analysis Summary:")
    print("=====================")
    print(df_metrics.groupby('experiment')[['near_-90_%', 'corr']].mean())


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate KNN localization performance")
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=RESULTS_DIR,
        help='Directory containing simulation results'
    )
    parser.add_argument(
        '--gt-path',
        type=Path,
        help='Path to ground truth data file'
    )
    parser.add_argument(
        '--single',
        type=Path,
        help='Evaluate a single result file'
    )
    parser.add_argument(
        '--analyze-rssi',
        action='store_true',
        help='Analyze RSSI statistics'
    )
    
    args = parser.parse_args()
    
    #  ground truth path
    if args.gt_path:
        gt_path = str(args.gt_path)
    else:
        # Look for ground truth file in results or data directory
        possible_paths = [
            args.results_dir / "rome_with_google_earth_locs.npy",
            DATA_DIR / "rome_with_google_earth_locs.npy",
        ]
        
        gt_path = None
        for path in possible_paths:
            if path.exists():
                gt_path = str(path)
                break
        
        if not gt_path:
            logger.error("Ground truth file not found. Please specify with --gt-path")
            return
    
    logger.info(f"Using ground truth: {gt_path}")
    
    #  evaluation
    if args.single:
        # Evaluate single file
        evaluate_single_result(str(args.single), gt_path)
    else:
        # Evaluate all results
        evaluate_all_results(args.results_dir, gt_path)
        
        if args.analyze_rssi:
            analyze_rssi_statistics(args.results_dir, gt_path)


if __name__ == "__main__":
    main()
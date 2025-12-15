"""
Sionna IEEE experiment modules.
"""
from .simulation import (
    clear_scene, 
    lonlat_to_local, 
    calculate_metrics, 
    run_simulation,
    evaluate_simulation
)
from .data_processing import (
    load_rome_data,
    correct_bs_coordinates,
    create_tx_rx_maps,
    create_data_arrays,
    BS_CORRECTIONS
)
from .scene_generation import (
    extract_height_from_osm,
    fetch_osm_buildings,
    convert_coordinates_to_local,
    create_building_ply,
    create_ground_plane_ply,
    generate_scene_xml
)
from .experiment_runner import (
    ExperimentRunner,
    DEFAULT_EXP_PARAMS,
    BEST_PER_BS_PARAMS
)
from .knn_evaluation import (
    knn_eval,
    analyze_rssi_experiments
)

__all__ = [
    # Simulation
    'clear_scene',
    'lonlat_to_local',
    'calculate_metrics',
    'run_simulation',
    'evaluate_simulation',
    # Data processing
    'load_rome_data',
    'correct_bs_coordinates',
    'create_tx_rx_maps',
    'create_data_arrays',
    'BS_CORRECTIONS',
    # Scene generation
    'extract_height_from_osm',
    'fetch_osm_buildings',
    'convert_coordinates_to_local',
    'create_building_ply',
    'create_ground_plane_ply',
    'generate_scene_xml',
    # Experiment runner
    'ExperimentRunner',
    'DEFAULT_EXP_PARAMS',
    'BEST_PER_BS_PARAMS',
    # KNN evaluation
    'knn_eval',
    'analyze_rssi_experiments',
]
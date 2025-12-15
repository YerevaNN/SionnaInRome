import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.rt.path_solvers import PathSolver
from pyproj import Transformer
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

# transformer for coordinate conversion
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)


def clear_scene(scene):
    for tx_id in scene.transmitters:
        scene.remove(tx_id)
    
    for rx_id in scene.receivers:
        scene.remove(rx_id)


def lonlat_to_local(lon, lat, center_lon, center_lat):
    x, y = transformer.transform(lon, lat)
    center_x, center_y = transformer.transform(center_lon, center_lat)
    return x - center_x, y - center_y


def calculate_metrics(paths, num_re=1008, num_rb=84, noise_power=-100.0, interference_power=-110.0):
    # paths.a shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    # sum over antenna and path dims to get [num_rx, num_tx]
    logger.debug("Calculating metrics")
    try:
        a_real, a_imag = paths.a
        # Convert to numpy arrays
        a_real_np = a_real.numpy()
        a_imag_np = a_imag.numpy()
        a_np = a_real_np + 1j * a_imag_np
        nd = a_np.ndim
        
        if nd == 5:
            # Sum over RX antenna (axis1), TX antenna (axis3) and paths (axis4)
            power = np.sum(np.abs(a_np)**2, axis=(1,3,4))
        elif nd == 3:
            # Already [num_rx, num_tx, num_paths]
            power = np.sum(np.abs(a_np)**2, axis=-1)
        else:
            # Otherwise, sum over the last axis
            power = np.sum(np.abs(a_np)**2, axis=-1)
        
        epsilon = 1e-9
        power = np.maximum(power, epsilon)
        rssi_dbm = 10 * np.log10(power)
        noise_linear = 10 ** (noise_power / 10.0)
        interference_linear = 10 ** (interference_power / 10.0)
        nsinr = power / (noise_linear + interference_linear)
        nsinr_db = 10 * np.log10(nsinr)
        nrsrp_power = power / num_re
        nrsrp_dbm = 10 * np.log10(nrsrp_power) + 30
        nrsrq = (num_rb * nrsrp_power) / power
        nrsrq_db = 10 * np.log10(nrsrq)
        logger.debug(f"Computed power shape: {power.shape}")
        return rssi_dbm, nsinr_db, nrsrp_dbm, nrsrq_db
    except Exception as e:
        logger.error(f"Error in calculate_metrics: {e}", exc_info=True)
        return None


def run_simulation(scene, tx_map, rx_map, exp_params, center_lon, center_lat, 
                   exp_params_per_bs=None):
    logger.info(f"Starting simulation: {exp_params['name']}")
    scene.frequency = exp_params['frequency']
    
    logger.info("Clearing the scene")
    clear_scene(scene)
    
    RX_ALTITUDE = 1.0  # UE antenna height (m)
    tx_order = []  # Track TX indices in the order they are added to the scene
    
    # Add transmitters
    logger.info("Adding transmitters")
    for loc, (tx_idx, tx_name, tx_lat, tx_lon) in tx_map.items():
        tx_order.append(tx_idx)
        # Get per-BS config if available
        bs_cfg = exp_params_per_bs.get(tx_idx, exp_params) if exp_params_per_bs else exp_params
        tx_altitude = bs_cfg.get('tx_height', exp_params['tx_height'])
        tx_orientation = bs_cfg.get('tx_orientation', exp_params['tx_orientation'])
        
        # Convert orientation to Sionna format: [alpha, beta, gamma] in radians
        # Default assumed order in configs: [azimuth_deg, elevation_deg (pitch), roll_deg]
        # Set exp_params["orientation_order"] to "roll_pitch_az" to use [roll, pitch, azimuth]
        if len(tx_orientation) == 3:
            orientation_order = bs_cfg.get("orientation_order", "az_el_roll")
            if orientation_order == "roll_pitch_az":
                roll_deg, pitch_deg, azimuth_deg = tx_orientation
            else:
                azimuth_deg, elevation_deg, roll_deg = tx_orientation
                pitch_deg = elevation_deg
            # Convert to radians for Sionna
            sionna_orientation = [
                float(np.radians(roll_deg)),       # alpha (roll)
                float(np.radians(pitch_deg)),      # beta (pitch/downtilt)
                float(np.radians(azimuth_deg))     # gamma (yaw/azimuth)
            ]
        else:
            sionna_orientation = [float(v) for v in tx_orientation]  # assume already in correct format
        
        tx_x, tx_y = lonlat_to_local(tx_lon, tx_lat, center_lon, center_lat)
        position_tuple = (float(tx_x), float(tx_y), float(tx_altitude))
        orientation_tuple = tuple(float(v) for v in sionna_orientation)
        try:
            scene.add(
                Transmitter(
                    name=tx_name,
                    position=position_tuple,
                    orientation=orientation_tuple,
                )
            )
        except Exception as e:
            logger.error(
                f"Failed to add transmitter '{tx_name}' with position={position_tuple} "
                f"and orientation(rad)={orientation_tuple}: {e}",
                exc_info=True,
            )
            raise
    
    # Add receivers
    logger.info("Adding receivers")
    for loc, (rx_idx, rx_name, rx_lat, rx_lon) in rx_map.items():
        # Get per-BS config if available (though typically not used for RX)
        bs_cfg = exp_params_per_bs.get(rx_idx, exp_params) if exp_params_per_bs else exp_params
        rx_orientation = bs_cfg.get('rx_orientation', exp_params['rx_orientation'])
        
        # Convert orientation to Sionna format: [alpha, beta, gamma] in radians
        # Default assumed order in configs: [azimuth_deg, elevation_deg (pitch), roll_deg]
        if len(rx_orientation) == 3:
            azimuth_deg, elevation_deg, roll_deg = rx_orientation
            pitch_deg = elevation_deg
            sionna_orientation = [
                float(np.radians(roll_deg)),
                float(np.radians(pitch_deg)),
                float(np.radians(azimuth_deg)),
            ]
        else:
            sionna_orientation = [float(v) for v in rx_orientation]  # assume already in correct format
        
        rx_x, rx_y = lonlat_to_local(rx_lon, rx_lat, center_lon, center_lat)
        rx_position = (float(rx_x), float(rx_y), float(RX_ALTITUDE))
        rx_orientation_tuple = tuple(float(v) for v in sionna_orientation)
        try:
            scene.add(
                Receiver(
                    name=rx_name,
                    position=rx_position,
                    orientation=rx_orientation_tuple,
                )
            )
        except Exception as e:
            logger.error(
                f"Failed to add receiver '{rx_name}' with position={rx_position} "
                f"and orientation(rad)={rx_orientation_tuple}: {e}",
                exc_info=True,
            )
            raise
    
    logger.info(f"{len(scene.transmitters)} TX and {len(scene.receivers)} RX added")
    
    # Configure antenna arrays
    scene.tx_array = PlanarArray(
        num_rows=exp_params["num_rows"],
        num_cols=exp_params["num_cols"],
        vertical_spacing=0.7,
        horizontal_spacing=0.5,
        pattern=exp_params["tx_pattern"],
        polarization=exp_params["tx_polarization"],
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=exp_params["rx_pattern"],
        polarization=exp_params["rx_polarization"],
    )
    
    # Run path solving
    try:
        logger.info("Computing paths with PathSolver")
        solver = PathSolver()
        paths = solver(
            scene=scene,
            max_depth=exp_params["max_depth"],
            max_num_paths_per_src=exp_params["max_num_paths_per_src"],
            samples_per_src=exp_params["samples_per_src"],
            synthetic_array=exp_params["synthetic_array"],
            los=exp_params["los"],
            specular_reflection=exp_params["specular_reflection"],
            diffuse_reflection=exp_params["diffuse_reflection"],
            refraction=exp_params["refraction"],
            seed=exp_params["seed"],
        )
    except Exception as e:
        logger.error(f"Error computing paths: {e}", exc_info=True)
        return None
    
    logger.info("Path computation done")
    
    # Calculate metrics
    metrics = calculate_metrics(paths)
    if metrics is None:
        logger.error("Failed to calculate metrics")
        return None

    rssi_dbm, nsinr_db, nrsrp_dbm, nrsrq_db = metrics

    # Apply per-BS transmit power adjustments in dB, if provided
    # Expected in configs: 'power_dbm' under exp_params or per-BS overrides
    try:
        base_power_dbm = float(exp_params.get('power_dbm', 0.0))
        if exp_params_per_bs or ('power_dbm' in exp_params):
            # Build vector of per-TX power dBm in the order transmitters were added
            per_tx_power_dbm = []
            for idx in tx_order:
                cfg = exp_params_per_bs.get(idx, exp_params) if exp_params_per_bs else exp_params
                per_tx_power_dbm.append(float(cfg.get('power_dbm', base_power_dbm)))
            per_tx_power_dbm = np.array(per_tx_power_dbm, dtype=float)
            # Use relative offset to base to avoid assuming an absolute baseline in the solver
            per_tx_offset_db = per_tx_power_dbm - base_power_dbm
            # Broadcast add across receivers dimension
            rssi_dbm = rssi_dbm + per_tx_offset_db[None, :]
            nsinr_db = nsinr_db + per_tx_offset_db[None, :]
            # NRSRP is also proportional to TX power
            nrsrp_dbm = nrsrp_dbm + per_tx_offset_db[None, :]
            # NRSRQ remains unchanged (ratio cancels power)
            logger.info(f"Applied TX power offsets (dB) per BS (order added): {per_tx_offset_db.tolist()}")
    except Exception as e:
        logger.error(f"Failed to apply power scaling: {e}")

    return rssi_dbm, nsinr_db, nrsrp_dbm, nrsrq_db


def evaluate_simulation(sionna_data, rome_data, tx_map):
    spearmans = []
    num_bs = len(tx_map)
    
    for tx_id in range(num_bs):
        # Extract RSSI values (first column)
        sim_rssi = sionna_data[tx_id, :, 0]
        real_rssi = rome_data[tx_id, :, 0]
        
        # Calculate Spearman correlation
        corr, p_value = spearmanr(real_rssi, sim_rssi)
        logger.info(f"Spearman for TX {tx_id}: {corr:.3f}")
        spearmans.append(corr)
    
    return spearmans
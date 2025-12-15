import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_rome_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Create location columns
    df['BS_location'] = df['BSLatitude'].round(6).astype(str) + '_' + df['BSLongitude'].round(6).astype(str)
    df['UE_location'] = df['UELatitude'].round(6).astype(str) + '_' + df['UELongitude'].round(6).astype(str)
    
    # Drop NaN values
    df = df.dropna(subset=['RSSI', 'BSLatitude', 'BSLongitude'])
    logger.info(f"Data shape after removing NaN values: {df.shape}")
    
    return df


def correct_bs_coordinates(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    # Apply Google Earth corrections: 10 BS locations -> 6 actual
    df = df.copy()
    
    # Map corrections
    mapped = df["BS_location"].map(mapping)
    tmp = pd.DataFrame(mapped.tolist(),
                      columns=["_lat_new", "_lon_new"],
                      index=df.index)
    
    # Update numeric columns
    df["BSLatitude"] = tmp["_lat_new"].combine_first(df["BSLatitude"])
    df["BSLongitude"] = tmp["_lon_new"].combine_first(df["BSLongitude"])
    
    # Update BS_location string where corrections were applied
    mask = tmp["_lat_new"].notna()
    df.loc[mask, "BS_location"] = (
        df.loc[mask, "BSLatitude"].round(8).astype(str) + "_" +
        df.loc[mask, "BSLongitude"].round(8).astype(str)
    )
    
    return df


def create_tx_rx_maps(df):
    tx_map = {}  # location: [idx, name, lat, lon]
    rx_map = {}
    tx_counter = 0
    rx_counter = 0
    
    for i, (_, row) in enumerate(df.iterrows()):
        # Process transmitters
        if row['BS_location'] not in tx_map:
            tx_idx = tx_counter
            tx_name = f"tx_{tx_idx}"
            tx_map[row['BS_location']] = (tx_idx, tx_name, row['BSLatitude'], row['BSLongitude'])
            tx_counter += 1
        
        # Process receivers
        if row['UE_location'] not in rx_map:
            rx_idx = rx_counter
            rx_name = f"rx_{rx_idx}"
            rx_map[row['UE_location']] = (rx_idx, rx_name, row['UELatitude'], row['UELongitude'])
            rx_counter += 1
    
    logger.info(f"Created {len(tx_map)} TX and {len(rx_map)} RX mappings")
    return tx_map, rx_map


def create_data_arrays(df, tx_map, rx_map):
    n_bs = len(tx_map)
    n_ue = len(rx_map)
    
    # Initialize with -90 (missing value sentinel)
    data_array = np.full((n_bs, n_ue, 6), -90.0)
    
    # Fill in measurement data
    for _, row in df.iterrows():
        tx_idx, _, _, _ = tx_map[row['BS_location']]
        rx_idx, _, _, _ = rx_map[row['UE_location']]
        data_array[tx_idx, rx_idx, :4] = [
            row["RSSI"], row["NSINR"], row["NRSRP"], row["NRSRQ"]
        ]
    
    # Fill in UE coordinates (same for all BSs)
    for ue_loc, (rx_idx, _, ue_lat, ue_lon) in rx_map.items():
        data_array[:, rx_idx, 4] = ue_lat
        data_array[:, rx_idx, 5] = ue_lon
    
    # Verify no missing coordinates
    assert not (data_array[:, :, 4] == -90).any(), "Missing UE latitudes"
    assert not (data_array[:, :, 5] == -90).any(), "Missing UE longitudes"
    
    return data_array


# Google Earth corrected BS locations
BS_CORRECTIONS = {
    "41.871766_12.461936": (41.87255347, 12.46204160),
    "41.868917_12.462432": (41.86982378, 12.46159222),
    "41.870385_12.46081":  (41.86982378, 12.46159222),
    "41.865419_12.465402": (41.86603287, 12.46527750),
    "41.869046_12.46551":  (41.86603287, 12.46527750),
    "41.871251_12.464301": (41.87255347, 12.46204160),
    "41.871046_12.471436": (41.87131243, 12.46805397),
    "41.873666_12.453575": (41.87290044, 12.45348311),
    "41.864758_12.469067": (41.86384904, 12.46890104),
    "41.866025_12.465944": (41.86603287, 12.46527750),
}
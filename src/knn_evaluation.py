import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import logging

logger = logging.getLogger(__name__)


def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000.0  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (np.sin(dlat / 2) ** 2 + 
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def mean_haversine(y_true, y_pred):
    return haversine(y_true[:, 0], y_true[:, 1],
                    y_pred[:, 0], y_pred[:, 1]).mean()


def knn_eval(sim_path: str,
             gt_path: str,
             k: int = 10,
             test_size: float = 0.10,
             random_state: int = 1,
             feature_index: int = 0,  # 0=RSSI, 1=NSINR, 2=NRSRP, 3=NRSRQ
             missing_sentinel: float = -90.0,
             feature_cols: list = None):
    """Run 4 K-NN experiments. Parameters
    ----------
    sim_path : str
        *.npy* file produced by Sionna, shape = (n_bs, n_ue, 4).
    gt_path  : str
        Ground-truth *.npy* with lat/lon, shape = (n_bs, n_ue, 6).
    k, test_size, random_state : usual sklearn parameters
    feature_index : int
        0 = RSSI, 1 = NSINR, 2 = NRSRP, 3 = NRSRQ.
    missing_sentinel : float
        Value indicating missing data
    feature_cols : list
        List of feature column names to use
    """
    # Load data
    gt = np.load(gt_path)    # (n_bs, n_ue, 6)
    sim = np.load(sim_path)  # (n_bs, n_ue, 4)
    n_bs, n_ue, _ = gt.shape

    # Feature frames
    def _array_to_df(arr, idx, prefix="BS"):
        feats = arr[:, :, idx]  # choose one feature → (n_bs, n_ue)
        return pd.DataFrame(feats.T,
                           columns=[f"{prefix}{i}" for i in range(n_bs)])

    df_real_feat = _array_to_df(gt, feature_index)
    df_sim_feat = _array_to_df(sim, feature_index)

    # Targets (take lat/lon from first-BS slice; identical for all BS)
    lat_lon = gt[0, :, 4:6]  # shape = (n_ue, 2)
    df_tgt = pd.DataFrame(lat_lon, columns=["ue_lat", "ue_lon"])

    # Concat features + targets
    df_real = pd.concat([df_real_feat, df_tgt], axis=1)
    df_sim = pd.concat([df_sim_feat, df_tgt], axis=1)

    # Drop rows with missing sentinel
    good = (df_tgt != missing_sentinel).all(axis=1)
    df_real, df_sim = df_real[good], df_sim[good]

    # Feature columns
    if feature_cols is None:
        feature_cols = [c for c in df_real.columns if c not in ("ue_lat", "ue_lon")]
    
    logger.info(f"Using features: {feature_cols}")
    target_cols = ["ue_lat", "ue_lon"]

    # Shared train/test split
    idx_train, idx_test = train_test_split(
        np.arange(len(df_real)),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    def _slice(df, idx, cols):
        return df.iloc[idx][cols].values

    def _run(X_train_df, X_test_df, label):
        X_train = _slice(X_train_df, idx_train, feature_cols)
        y_train = _slice(df_real, idx_train, target_cols)  # always real lat/lon
        X_test = _slice(X_test_df, idx_test, feature_cols)
        y_test = _slice(df_real, idx_test, target_cols)

        knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        
        return {
            "Setting": label,
            "K": k,
            "Test pts": len(idx_test),
            "Mean Haversine (m)": round(mean_haversine(y_test, pred), 2),
        }

    # 4 scenarios
    results = [
        _run(df_real, df_real, "Train REAL → Test REAL"),
        _run(df_real, df_sim, "Train REAL → Test SIM"),
        _run(df_sim, df_sim, "Train SIM → Test SIM"),
        _run(df_sim, df_real, "Train SIM → Test REAL"),
    ]

    return pd.DataFrame(results)


def analyze_rssi_experiments(npy_files, real_data_path, threshold=0.2):
    """
    Analyze RSSI experiments for -90 dB values and correlations.
    
    Args:
        npy_files: List of simulation result files
        real_data_path: Path to ground truth data
        threshold: Threshold for considering values near -90 dB
    
    Returns:
        DataFrame with analysis results
    """
    real = np.load(real_data_path)
    rows = []

    for f in npy_files:
        sim = np.load(f)
        cnt, pct, match, corr = _summarize_single(sim, real, threshold)
        for bs in range(sim.shape[0]):
            rows.append({
                "experiment": os.path.basename(f),
                "BS": bs,
                "near_-90_cnt": cnt[bs],
                "near_-90_%": pct[bs],
                "matches": match[bs],
                "corr": corr[bs]
            })

    df = pd.DataFrame(rows)
    return df


def _summarize_single(sim_data: np.ndarray,
                     real_data: np.ndarray,
                     threshold: float = 0.2):
    """
    Compute per-BS statistics for one simulated tensor.
    Assumes shape (n_bs, n_ue, ≥1) and that column-0 is RSSI.
    """
    sim = sim_data[:, :, 0]   # (BS, UE)
    real = real_data[:, :, 0]

    near_sim = np.abs(sim + 90) <= threshold     # bool mask
    near_real = np.abs(real + 90) <= threshold
    both_near = near_sim & near_real

    counts = near_sim.sum(axis=1)
    pct = counts / sim.shape[1] * 100
    matches = both_near.sum(axis=1)
    corrs = [np.corrcoef(sim[b], real[b])[0, 1] for b in range(sim.shape[0])]

    return counts, pct, matches, corrs
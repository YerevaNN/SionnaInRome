# Guide will help you reproduce the experiments from https://arxiv.org/abs/2507.19653 with proper OSM building heights.
 
## Prerequisites

### 1. Install Dependencies
```bash
cd sionna_ieee
pip install -r requirements.txt
```

### 2. Copy Dataset
Copy the Rome dataset CSV file to the data directory:
```bash
cp /path/to/rome_crop_knn_ready.csv sionna_ieee/data/
```

## Running the Complete Workflow

### Step 1: Generate Scene with Proper Building Heights
```bash
cd sionna_ieee
python generate_scene.py
```

This will:
- Fetch building data from OpenStreetMap for coordinates (41.861824 to 41.8778842, 12.4525696 to 12.471929)
- Extract proper building heights using multiple methods:
  - Direct height attributes from OSM
  - Building levels with appropriate floor heights  
  - Building type-specific defaults with realistic variation
- Generate PLY mesh files for all buildings
- Create the Sionna scene XML file at `scenes/rome_scene_with_heights.xml`


### Step 2: Run Baseline Experiment
```bash
python run_experiments.py --experiment baseline
```

### Step 3: Run Optimized Per-BS Experiment
```bash
python run_experiments.py --experiment optimized
```


### Step 4: Evaluate KNN Localization Performance
```bash
python evaluate_knn.py
```

### Step 5 : Run All Parameter Sweeps
```bash
python run_experiments.py --experiment sweeps
```

This runs comprehensive parameter sweeps covering:
- TX heights: [11, 12, 15, 20, 40, 55]m
- Frequencies: [1.01, 1.05, 1.1, 1.2, 1.3, 1.5, 2, 3.6, 5]GHz
- TX/RX patterns: ["iso", "dipole", "hw_dipole", "tr38901"]
- Ray tracing depths: [1, 2, 3, 4, 5, 6, 7]
- TX/RX azimuth orientations: 0° to 330° in 30° steps


## File Outputs

All results are saved in the `results/` directory:
- `*.npy`: Simulation result tensors (shape: [n_bs, n_ue, 4] for RSSI, NSINR, NRSRP, NRSRQ)
- `*_config.npy`: Experiment configuration files
- `knn_evaluation_summary.csv`: KNN localization results
- `rssi_analysis_summary.csv`: RSSI statistics analysis



The framework now properly reproduces the paper experiments with realistic building heights extracted from OpenStreetMap data.

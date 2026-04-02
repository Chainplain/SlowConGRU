# QuatLearner: GRU-Based Quaternion and Angular-Rate Slow Component Learning

## Project Title
QuatLearner: Curriculum GRU for quaternion and omega slow-component extraction for real flapping-wing flight data.

## Flapping wing flight data
You can find the realflight data here:
https://drive.google.com/file/d/14RCxahoPVs62AJNMx4_9b1vWLCO5-nuo/view

## Description
This project trains and uses a GRU model to estimate slow components of:
- Quaternion attitude: q_slow = [w, x, y, z]
- Angular velocity: omega_slow = [x, y, z]

The main workflow is:
1. Train with curriculum and overlapping windows.
2. Save best model and normalization artifacts.
3. Run step-by-step autoregressive inference with hidden state carry-over.
4. Plot and analyze training/prediction results.

Main scripts:
- gru_q_omega_vec_slow.py: training script (model, losses, curriculum, export CSV/figures/checkpoint)
- load_GRU_and_gen.py: load trained network and run inference on a CSV
- plotGRU_result.py: replot figures from saved CSV outputs (no retraining)

## Demo / Screenshots
After running the scripts, typical outputs are generated in training_figures_and_csv and training_figures_and_csv/replotted_figures, such as:
- fig1_training_convergence.png
- fig2_error_metrics.svg
- fig3a_prediction_quaternion.png
- fig3b_prediction_omega.png
- load_gru_prediction_vs_input.png

You can view existing figures in:
- training_figures_and_csv
- training_figures_and_csv/replotted_figures
## Preprocessing
### Overview
Before training, you must generate slow components from raw flight data.

### Script: find_slow_comp.py
Run:

```powershell
python find_slow_comp.py
```

This script:
- Loads raw segment CSV files (q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z)
- Applies low-pass filtering to extract slow components
- Generates slow_q and slow_omega data
- Outputs preprocessed CSVs ready for training

Output CSVs are used as targets by gru_q_omega_vec_slow.py during the training workflow.

## Features
- Curriculum learning over sequence length and stride.
- Overlapping chunk dataset generation for better data coverage.
- Joint prediction heads for quaternion and omega slow components.
- Quaternion-safe handling:
  - normalization
  - hemisphere canonicalization
- Kinematic consistency loss linking quaternion trajectory and angular rate.
- Step-by-step recurrent inference with hidden-state propagation.
- Export of:
  - best checkpoint
  - normalization stats
  - training history CSVs
  - prediction/error CSVs
  - publication-style plots

## Installation
### 1) Python environment
Use your existing environment in this folder (env) or create one:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
pip install numpy pandas matplotlib torch
```

### 3) Verify key files exist
- segment_data_0204.csv
- segment_data_1227.csv
- segment_data_1228_1.csv
- segment_data_1229.csv

## Usage
### A) Train the network
Run:

```powershell
python gru_q_omega_vec_slow.py
```

This script will:
- train the GRU with curriculum windows
- save best weights to:
  - training_figures_and_csv/gru_qslow_omegaslow_best_curriculum_overlap_kin.pt
- save omega normalization to:
  - training_figures_and_csv/omega_normalization.npy
- save training/prediction CSVs and figures in training_figures_and_csv

### B) Load the network and run inference
Run:

```powershell
python load_GRU_and_gen.py
```

Default behavior:
- Loads checkpoint:
  - training_figures_and_csv/gru_qslow_omegaslow_best_curriculum_overlap_kin.pt
- Loads normalization:
  - training_figures_and_csv/omega_normalization.npy
- Runs inference on:
  - segment_data_1228_1.csv
- Saves comparison plot:
  - training_figures_and_csv/load_gru_prediction_vs_input.png

Optional arguments:

```powershell
python load_GRU_and_gen.py \
  --checkpoint training_figures_and_csv/gru_qslow_omegaslow_best_curriculum_overlap_kin.pt \
  --omega-norm training_figures_and_csv/omega_normalization.npy \
  --input-csv segment_data_1228_1.csv \
  --output-plot training_figures_and_csv/load_gru_prediction_vs_input.png
```

How to use the net in practice:
1. Prepare an input CSV with columns:
   - q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z
2. Load model weights and omega normalization.
3. Build normalized input X_norm from model input_mean and input_std.
4. Predict sequentially with hidden-state carry-over.
5. Denormalize omega prediction with omega_std and omega_mean.
6. Plot predicted slow components against raw inputs.

### C) Replot result figures from CSV files only
Run:

```powershell
python plotGRU_result.py
```

This script expects these CSVs in training_figures_and_csv:
- training_history.csv
- prediction_timeseries.csv
- prediction_errors.csv
- prediction_summary_statistics.csv
- optional: training_phase_summary.csv

It writes fresh figures into:
- training_figures_and_csv/replotted_figures

## Third party libs
- numpy
- pandas
- torch (PyTorch)
- matplotlib

## Notes
- If a script reports missing file errors, run training first or check path names.
- On Windows, script names are case-insensitive at runtime, but keep consistent naming in docs and commits.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from logmap_demon import quat_to_rotvec_log

# -----------------------
# Reproducibility
# -----------------------
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# Config
# -----------------------
SEG_LEN = 200
DATA_FILES = [
    "segment_data_0204.csv",
    "segment_data_1227.csv",
    "segment_data_1228_1.csv",
    "segment_data_1229.csv",
]

# -----------------------
# Data loading
# -----------------------
def load_file(path):
    df = pd.read_csv(path)

    q_array      = df[["q_w","q_x","q_y","q_z"]].to_numpy()
    q_slow_array = df[["q_slow_w","q_slow_x","q_slow_y","q_slow_z"]].to_numpy()
    omega        = df[["omega_x","omega_y","omega_z"]].to_numpy()
    omega_slow   = df[["omega_slow_x", "omega_slow_y", "omega_slow_z"]].to_numpy()

    r      = np.array([quat_to_rotvec_log(q) for q in q_array], dtype=np.float32)
    r_slow = np.array([quat_to_rotvec_log(q) for q in q_slow_array], dtype=np.float32)

    omega = omega.astype(np.float32)
    omega_slow = omega_slow.astype(np.float32)

    return r, r_slow, omega, omega_slow

def make_chunks(r, r_slow, omega, omega_slow, seg_len):
    inputs  = np.concatenate([r, omega], axis=1).astype(np.float32)   # (N,6)
    targets = np.concatenate([r_slow, omega_slow], axis=1).astype(np.float32)  # (N,6)

    n_chunks = len(inputs) // seg_len
    X_chunks, Y_chunks = [], []

    for i in range(n_chunks):
        s = i * seg_len
        X_chunks.append(inputs[s:s+seg_len])
        Y_chunks.append(targets[s:s+seg_len])

    return X_chunks, Y_chunks

all_X, all_Y = [], []

for path in DATA_FILES:
    print(f"Loading {path} ...")
    r, r_slow, omega, omega_slow = load_file(path)
    X_chunks, Y_chunks = make_chunks(r, r_slow, omega, omega_slow, SEG_LEN)

    all_X.extend(X_chunks)
    all_Y.extend(Y_chunks)

    print(f"  -> {len(X_chunks)} chunks  ({len(r)} samples)")

all_X = np.stack(all_X, axis=0).astype(np.float32)
all_Y = np.stack(all_Y, axis=0).astype(np.float32)

print(f"\nTotal chunks: {len(all_X)}  shape X={all_X.shape}  Y={all_Y.shape}")

# -----------------------
# Normalize
# -----------------------
x_mean = all_X.mean(axis=(0,1), keepdims=True)
x_std  = all_X.std(axis=(0,1), keepdims=True) + 1e-6

y_mean = all_Y.mean(axis=(0,1), keepdims=True)
y_std  = all_Y.std(axis=(0,1), keepdims=True) + 1e-6

# -----------------------
# Model definition
# -----------------------
class GRUSlowExtractor(nn.Module):
    """
    Predicts [r_slow, omega_slow] from input [r, omega].
    Adds residual paths at every recurrent layer and finally from r -> output.
    """

    def __init__(self, input_size=6, output_size=6, hidden_size=64, num_layers=3, dropout=0.1):
        super().__init__()

        assert input_size >= 3, "Input must contain r in the first 3 dimensions for skip-add."

        self.output_size = output_size  # 6
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for layer_idx in range(num_layers):
            in_size = input_size if layer_idx == 0 else hidden_size

            self.layers.append(
                nn.GRU(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True
                )
            )

            if in_size == hidden_size:
                self.residual_projs.append(nn.Identity())
            else:
                self.residual_projs.append(nn.Linear(in_size, hidden_size))

        # Head predicts delta in output space
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, h0=None):
        layer_input = x
        hT_list = []

        for layer_idx, (gru_layer, res_proj) in enumerate(zip(self.layers, self.residual_projs)):

            h_init = None
            if h0 is not None:
                h_init = h0[layer_idx:layer_idx + 1]

            z, h_layer = gru_layer(layer_input, h_init)  # (B,T,H)
            hT_list.append(h_layer)

            residual = res_proj(layer_input)
            z = z + residual                           # layer-wise skip

            if layer_idx < self.num_layers - 1:
                z = self.dropout_layer(z)

            layer_input = z

        hT = torch.cat(hT_list, dim=0) if hT_list else None

        delta = self.head(layer_input)                 # (B,T,6)

        # Skip connection from input r -> output r_slow part only.
        # omega_slow part is learned from recurrent features.
        r_in = x[:, :, :3]                             # (B,T,3)
        y = delta
        y[:, :, :3] = y[:, :, :3] + r_in

        return y, hT

# -----------------------
# Load model weights
# -----------------------
model = GRUSlowExtractor().to(device)
state = torch.load("gru_rotvec_slow_best.pt", map_location=device)
model.load_state_dict(state["model"] if "model" in state else state)

# -----------------------
# Step-by-step prediction
# -----------------------
print("\nStep-by-step prediction on segment_data_1228_1")

r, r_slow, omega, omega_slow = load_file("segment_data_1228_1.csv")

inputs = np.concatenate([r, omega], axis=1).astype(np.float32)
targets = np.concatenate([r_slow, omega_slow], axis=1).astype(np.float32)

X_norm = ((inputs - x_mean.squeeze()) / x_std.squeeze()).astype(np.float32)
Y_true = targets

model.eval()
predictions = []
hidden_state = None

with torch.no_grad():
    for t in range(len(X_norm)):
        x_t = torch.from_numpy(X_norm[t:t+1]).float().unsqueeze(0).to(device)  # (1,1,6)
        pred_norm, hidden_state = model(x_t, hidden_state)
        if hidden_state is not None:
            hidden_state = hidden_state.detach()
        pred_norm = pred_norm.squeeze(0).cpu().numpy()  # (1,6)
        pred_denorm = pred_norm * y_std.squeeze() + y_mean.squeeze()
        predictions.append(pred_denorm[0])

predictions = np.array(predictions)

# -----------------------
# Filter Predictions
# -----------------------
def smooth_predictions(predictions, window_size=10):
    kernel = np.ones(window_size) / window_size
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=predictions)
    return smoothed

# Apply smoothing to predictions
# predictions_smoothed = smooth_predictions(predictions)
predictions_smoothed = predictions  # No smoothing for now

# -----------------------
# Plot
# -----------------------
axis_labels = ["x", "y", "z"]
t_steps = np.arange(len(Y_true))

q_slow_true = Y_true[:, :3]
q_slow_pred = predictions_smoothed[:, :3]
omega_slow_true = Y_true[:, 3:]
omega_slow_pred = predictions_smoothed[:, 3:]

fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True)

for i in range(3):
    axes[i, 0].plot(t_steps, q_slow_true[:, i], label="q_slow", linewidth=2, color="#336ad8")
    axes[i, 0].plot(t_steps, q_slow_pred[:, i], label="q_slow_predicted", linestyle="--", color="#de2040", zorder=3)
    axes[i, 0].plot(t_steps, r[:, i], label="q (original)",  color="#9BCAF7", linewidth=1.5)
    axes[i, 0].set_ylabel(axis_labels[i])
    axes[i, 0].legend()
    axes[i, 0].grid(True)

    axes[i, 1].plot(t_steps, omega_slow_true[:, i], label="omega_slow", linewidth=2, color="#2f7d4a")
    axes[i, 1].plot(t_steps, omega_slow_pred[:, i], label="omega_slow_predicted", linestyle="--", color="#de2040", zorder=3)
    axes[i, 1].plot(t_steps, omega[:, i], label="omega (original)",  color="#ADD190", linewidth=1.5)
    axes[i, 1].set_ylabel(axis_labels[i])
    axes[i, 1].legend()
    axes[i, 1].grid(True)

axes[0, 0].set_title("q_slow vs q_slow_predicted (rotvec)")
axes[0, 1].set_title("omega_slow vs omega_slow_predicted")

axes[-1, 0].set_xlabel("Time step")
axes[-1, 1].set_xlabel("Time step")

plt.tight_layout()
plt.show()

# -----------------------
# Metrics
# -----------------------
mse = np.mean((predictions - Y_true) ** 2)
rmse = np.sqrt(mse)

mse_r = np.mean((predictions[:, :3] - Y_true[:, :3]) ** 2)
rmse_r = np.sqrt(mse_r)

mse_omega = np.mean((predictions[:, 3:] - Y_true[:, 3:]) ** 2)
rmse_omega = np.sqrt(mse_omega)

print("MSE:", mse)
print("RMSE:", rmse)
print("MSE_r:", mse_r)
print("RMSE_r:", rmse_r)
print("MSE_omega:", mse_omega)
print("RMSE_omega:", rmse_omega)

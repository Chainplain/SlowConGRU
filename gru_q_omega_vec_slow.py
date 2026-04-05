import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
DATA_FILES = [
    "segment_data_0204.csv",
    "segment_data_1227.csv",
    "segment_data_1228_1.csv",
    "segment_data_1229.csv",
]

# Curriculum: (start_epoch, end_epoch, seg_len, stride)
CURRICULUM = [
    (1, 20, 50, 10),
    (21, 50, 100, 20),
    (51, 100, 200, 50),
]

BATCH_SIZE = 32
NUM_EPOCHS = 100
VAL_RATIO = 0.2

# ~180 Hz
DT = 1.0 / 180.0

# Loss weights
LAMBDA_Q = 1.0
LAMBDA_OMEGA = 1.0
LAMBDA_UNITY = 0.1
LAMBDA_KIN = 1.0

# Output folder
OUT_DIR = "training_figures_and_csv"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Theme colors (teal / deep blue family)
# -----------------------
C_TEAL = "#138D90"
C_TEAL_LIGHT = "#7FD3D4"
C_BLUE = "#1F4E79"
C_BLUE_LIGHT = "#7AA6D1"
C_NAVY = "#0B2545"
C_GRID = "#D9E6F2"
C_GRAY = "#6C7A89"
PHASE_COLORS = ["#D8F0F0", "#DCE8F8", "#EAF2FB"]

FIGSIZE = (10, 6)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35
plt.rcParams["grid.color"] = C_GRID

# -----------------------
# Quaternion utilities
# -----------------------
def normalize_quaternion_array(q):
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    return q / norm

def canonicalize_quaternion_array(q):
    """
    Force scalar part q[:, 0] >= 0 so all quaternions stay
    in the same hemisphere.
    """
    q = q.copy()
    flip_mask = q[:, 0] < 0.0
    q[flip_mask] *= -1.0
    return q

def canonicalize_quaternion_tensor(q):
    """
    q: (..., 4)
    Force scalar part q[..., 0] >= 0.
    """
    sign = torch.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q * sign

def quat_mul(q1, q2):
    """
    Hamilton product.
    q1, q2: (..., 4) with format [w, x, y, z]
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=-1)

def omega_to_quat(omega):
    """
    omega: (..., 3) -> pure quaternion (..., 4)
    """
    zeros = torch.zeros_like(omega[..., :1])
    return torch.cat([zeros, omega], dim=-1)

def integrate_quaternion_forward(q, omega, dt):
    """
    Discrete quaternion kinematics using forward Euler:
        q_dot = 0.5 * q ⊗ [0, omega]
        q_next = normalize(q + dt * q_dot)

    q:     (..., 4)
    omega: (..., 3)
    """
    omega_q = omega_to_quat(omega)
    q_dot = 0.5 * quat_mul(q, omega_q)
    q_next = q + dt * q_dot
    q_next = q_next / (torch.norm(q_next, dim=-1, keepdim=True) + 1e-8)
    q_next = canonicalize_quaternion_tensor(q_next)
    return q_next

# -----------------------
# Data loading
# -----------------------
def load_file(path):
    df = pd.read_csv(path)

    q_array = df[["q_w", "q_x", "q_y", "q_z"]].to_numpy(dtype=np.float32)
    q_slow_array = df[["q_slow_w", "q_slow_x", "q_slow_y", "q_slow_z"]].to_numpy(dtype=np.float32)

    omega = df[["omega_x", "omega_y", "omega_z"]].to_numpy(dtype=np.float32)
    omega_slow = df[["omega_slow_x", "omega_slow_y", "omega_slow_z"]].to_numpy(dtype=np.float32)

    q_array = normalize_quaternion_array(q_array)
    q_slow_array = normalize_quaternion_array(q_slow_array)

    q_array = canonicalize_quaternion_array(q_array)
    q_slow_array = canonicalize_quaternion_array(q_slow_array)

    return q_array, q_slow_array, omega.astype(np.float32), omega_slow.astype(np.float32)

# Cache raw file data once
raw_data = {}
for path in DATA_FILES:
    print(f"Loading raw {path} ...")
    q, q_slow, omega, omega_slow = load_file(path)
    raw_data[path] = {
        "q": q,
        "q_slow": q_slow,
        "omega": omega,
        "omega_slow": omega_slow,
    }
    print(f"  -> {len(q)} samples")

# -----------------------
# Global input normalization stats
# -----------------------
all_inputs_for_stats = []
for path in DATA_FILES:
    q = raw_data[path]["q"]
    omega = raw_data[path]["omega"]
    inp = np.concatenate([q, omega], axis=1).astype(np.float32)
    all_inputs_for_stats.append(inp)

all_inputs_for_stats = np.concatenate(all_inputs_for_stats, axis=0)
x_mean = all_inputs_for_stats.mean(axis=0, keepdims=True).reshape(1, 1, 7).astype(np.float32)
x_std = (all_inputs_for_stats.std(axis=0, keepdims=True) + 1e-6).reshape(1, 1, 7).astype(np.float32)

print("\nGlobal normalization stats ready.")

# -----------------------
# Overlapping chunk builder
# -----------------------
def make_chunks(q, q_slow, omega, omega_slow, seg_len, stride):
    """
    Build overlapping windows.

    Inputs:
        q:           (N, 4)
        q_slow:      (N, 4)
        omega:       (N, 3)
        omega_slow:  (N, 3)

    Returns:
        X_chunks: list of (seg_len, 7)
        Yq_chunks: list of (seg_len, 4)
        Yw_chunks: list of (seg_len, 3)
    """
    assert seg_len > 0
    assert stride > 0

    inputs = np.concatenate([q, omega], axis=1).astype(np.float32)   # (N, 7)
    target_q = q_slow.astype(np.float32)                             # (N, 4)
    target_w = omega_slow.astype(np.float32)                         # (N, 3)

    N = len(inputs)
    X_chunks, Yq_chunks, Yw_chunks = [], [], []

    if N < seg_len:
        return X_chunks, Yq_chunks, Yw_chunks

    for s in range(0, N - seg_len + 1, stride):
        X_chunks.append(inputs[s:s + seg_len])
        Yq_chunks.append(target_q[s:s + seg_len])
        Yw_chunks.append(target_w[s:s + seg_len])

    return X_chunks, Yq_chunks, Yw_chunks

def build_chunked_dataset(seg_len, stride):
    all_X, all_Yq, all_Yw = [], [], []

    for path in DATA_FILES:
        q = raw_data[path]["q"]
        q_slow = raw_data[path]["q_slow"]
        omega = raw_data[path]["omega"]
        omega_slow = raw_data[path]["omega_slow"]

        X_chunks, Yq_chunks, Yw_chunks = make_chunks(q, q_slow, omega, omega_slow, seg_len, stride)
        all_X.extend(X_chunks)
        all_Yq.extend(Yq_chunks)
        all_Yw.extend(Yw_chunks)

    all_X = np.stack(all_X, axis=0).astype(np.float32)
    all_Yq = np.stack(all_Yq, axis=0).astype(np.float32)
    all_Yw = np.stack(all_Yw, axis=0).astype(np.float32)

    Xn = ((all_X - x_mean) / x_std).astype(np.float32)

    # Keep quaternion target raw unit quaternion
    Yq = all_Yq.copy()

    # Normalize omega_slow target for regression stability
    omega_mean = all_Yw.mean(axis=(0, 1), keepdims=True)
    omega_std = all_Yw.std(axis=(0, 1), keepdims=True) + 1e-6
    Yw = ((all_Yw - omega_mean) / omega_std).astype(np.float32)

    return Xn, Yq, Yw, omega_mean.astype(np.float32), omega_std.astype(np.float32)

# -----------------------
# Dataset
# -----------------------
class QuaternionOmegaDataset(Dataset):
    def __init__(self, X, Yq, Yw):
        self.X = torch.from_numpy(X).float()
        self.Yq = torch.from_numpy(Yq).float()
        self.Yw = torch.from_numpy(Yw).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Yq[idx], self.Yw[idx]

def make_loaders_for_window(seg_len, stride, batch_size=BATCH_SIZE, val_ratio=VAL_RATIO):
    Xn, Yq, Yw, omega_mean, omega_std = build_chunked_dataset(seg_len, stride)

    n_total = len(Xn)
    n_val = max(1, int(val_ratio * n_total))

    idx = np.random.permutation(n_total)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = QuaternionOmegaDataset(Xn[train_idx], Yq[train_idx], Yw[train_idx])
    val_ds = QuaternionOmegaDataset(Xn[val_idx], Yq[val_idx], Yw[val_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"seg_len={seg_len}, stride={stride} | total chunks={n_total} | "
        f"train={len(train_idx)} | val={len(val_idx)}"
    )

    return train_loader, val_loader, omega_mean, omega_std

# -----------------------
# GRU Model
# -----------------------
class GRUSlowStateExtractor(nn.Module):
    """
    Predict both:
        q_slow_pre
        omega_slow_pre

    Input:
        x: normalized [q(4), omega(3)]

    Output:
        q_pred_raw:   raw quaternion before normalization
        q_pred:       normalized quaternion with scalar part >= 0
        omega_pred_n: normalized omega_slow prediction
    """

    def __init__(self, input_mean, input_std, input_size=7,
                 q_output_size=4, w_output_size=3,
                 hidden_size=64, num_layers=3, dropout=0.1):
        super().__init__()

        assert input_size == 7
        assert q_output_size == 4
        assert w_output_size == 3

        self.q_output_size = q_output_size
        self.w_output_size = w_output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.register_buffer("input_mean", torch.tensor(input_mean.squeeze(), dtype=torch.float32))
        self.register_buffer("input_std", torch.tensor(input_std.squeeze(), dtype=torch.float32))

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

        self.q_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, q_output_size)
        )

        self.w_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, w_output_size)
        )

        for name, p in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x, h0=None):
        layer_input = x
        hT_list = []

        for layer_idx, (gru_layer, res_proj) in enumerate(zip(self.layers, self.residual_projs)):
            h_init = None
            if h0 is not None:
                h_init = h0[layer_idx:layer_idx + 1]

            z, h_layer = gru_layer(layer_input, h_init)
            hT_list.append(h_layer)

            residual = res_proj(layer_input)
            z = z + residual

            if layer_idx < self.num_layers - 1:
                z = self.dropout_layer(z)

            layer_input = z

        hT = torch.cat(hT_list, dim=0) if hT_list else None

        delta_q = self.q_head(layer_input)       # (B, T, 4)
        omega_pred_n = self.w_head(layer_input)  # (B, T, 3)

        # Recover original input quaternion from normalized input
        q_in_norm = x[:, :, :4]
        q_mean = self.input_mean[:4].view(1, 1, 4)
        q_std = self.input_std[:4].view(1, 1, 4)

        q_in = q_in_norm * q_std + q_mean
        q_in = q_in / (torch.norm(q_in, dim=-1, keepdim=True) + 1e-8)
        q_in = canonicalize_quaternion_tensor(q_in)

        q_pred_raw = q_in + delta_q
        q_pred = q_pred_raw / (torch.norm(q_pred_raw, dim=-1, keepdim=True) + 1e-8)
        q_pred = canonicalize_quaternion_tensor(q_pred)

        return q_pred_raw, q_pred, omega_pred_n, hT

# -----------------------
# Losses
# -----------------------
class QuaternionGeodesicLoss(nn.Module):
    """
    loss = mean(1 - <q_pred, q_true>^2)
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, q_pred, q_true):
        q_pred = q_pred / (torch.norm(q_pred, dim=-1, keepdim=True) + self.eps)
        q_true = q_true / (torch.norm(q_true, dim=-1, keepdim=True) + self.eps)

        dot = torch.sum(q_pred * q_true, dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)

        loss = 1.0 - dot * dot
        return loss.mean()

class UnityLoss(nn.Module):
    """
    Encourage raw quaternion output to have unit norm before explicit normalization.
    """
    def forward(self, q_raw):
        q_norm = torch.norm(q_raw, dim=-1)
        return ((q_norm - 1.0) ** 2).mean()

class KinematicLoss(nn.Module):
    """
    Enforce consistency between q_pred and omega_pred via quaternion kinematics.
    """
    def __init__(self, dt=DT):
        super().__init__()
        self.dt = dt

    def forward(self, q_pred, omega_pred):
        if q_pred.shape[1] < 2:
            return torch.tensor(0.0, device=q_pred.device, dtype=q_pred.dtype)

        q_t = q_pred[:, :-1, :]
        omega_t = omega_pred[:, :-1, :]
        q_next_pred = q_pred[:, 1:, :]

        q_next_kin = integrate_quaternion_forward(q_t, omega_t, self.dt)

        dot = torch.sum(q_next_kin * q_next_pred, dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)

        loss = 1.0 - dot * dot
        return loss.mean()

# -----------------------
# Model / optimizer
# -----------------------
model = GRUSlowStateExtractor(
    input_mean=x_mean,
    input_std=x_std,
    input_size=7,
    q_output_size=4,
    w_output_size=3,
    hidden_size=64,
    num_layers=3,
    dropout=0.1
).to(device)

criterion_q = QuaternionGeodesicLoss()
criterion_w = nn.MSELoss()
criterion_unity = UnityLoss()
criterion_kin = KinematicLoss(dt=DT)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# -----------------------
# Evaluation helpers
# -----------------------
def quaternion_angle_error_deg_tensor(q_pred, q_true):
    q_pred = q_pred / (torch.norm(q_pred, dim=-1, keepdim=True) + 1e-8)
    q_true = q_true / (torch.norm(q_true, dim=-1, keepdim=True) + 1e-8)

    dot = torch.sum(q_pred * q_true, dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)

    angle = 2.0 * torch.acos(dot)
    angle_deg = angle * 180.0 / np.pi
    return angle_deg

def quaternion_angle_error_deg(q_pred, q_true):
    return quaternion_angle_error_deg_tensor(q_pred, q_true).mean().item()

def compute_total_loss(q_pred_raw, q_pred, omega_pred_n, yq_true, yw_true_n, omega_mean_t, omega_std_t):
    loss_q = criterion_q(q_pred, yq_true)
    loss_w = criterion_w(omega_pred_n, yw_true_n)
    loss_unity = criterion_unity(q_pred_raw)

    omega_pred = omega_pred_n * omega_std_t + omega_mean_t
    loss_kin = criterion_kin(q_pred, omega_pred)

    total_loss = (
        LAMBDA_Q * loss_q
        + LAMBDA_OMEGA * loss_w
        + LAMBDA_UNITY * loss_unity
        + LAMBDA_KIN * loss_kin
    )

    return total_loss, loss_q, loss_w, loss_unity, loss_kin

def eval_full(loader, omega_mean_t, omega_std_t):
    model.eval()

    sums = {
        "total": 0.0,
        "q": 0.0,
        "w": 0.0,
        "unity": 0.0,
        "kin": 0.0,
        "angle_deg": 0.0,
    }
    n = 0

    with torch.no_grad():
        for xb, yqb, ywb in loader:
            xb = xb.to(device, non_blocking=True)
            yqb = yqb.to(device, non_blocking=True)
            ywb = ywb.to(device, non_blocking=True)

            q_pred_raw, q_pred, omega_pred_n, _ = model(xb)
            total_loss, loss_q, loss_w, loss_unity, loss_kin = compute_total_loss(
                q_pred_raw, q_pred, omega_pred_n, yqb, ywb, omega_mean_t, omega_std_t
            )

            ang = quaternion_angle_error_deg(q_pred, yqb)

            bs = xb.size(0)
            sums["total"] += total_loss.item() * bs
            sums["q"] += loss_q.item() * bs
            sums["w"] += loss_w.item() * bs
            sums["unity"] += loss_unity.item() * bs
            sums["kin"] += loss_kin.item() * bs
            sums["angle_deg"] += ang * bs
            n += bs

    for k in sums:
        sums[k] /= max(n, 1)

    return sums

def get_window_config_for_epoch(epoch):
    for start_ep, end_ep, seg_len, stride in CURRICULUM:
        if start_ep <= epoch <= end_ep:
            return seg_len, stride
    _, _, seg_len, stride = CURRICULUM[-1]
    return seg_len, stride

# -----------------------
# Plot helpers
# -----------------------
def add_curriculum_phase_shading(ax, curriculum):
    for i, (start_ep, end_ep, seg_len, stride) in enumerate(curriculum):
        ax.axvspan(start_ep, end_ep, color=PHASE_COLORS[i % len(PHASE_COLORS)], alpha=0.45, zorder=0)
        xmid = 0.5 * (start_ep + end_ep)
        ymin, ymax = ax.get_ylim()
        ytxt = ymin + 0.92 * (ymax - ymin)
        ax.text(
            xmid, ytxt,
            f"Phase {i+1}\nL={seg_len}, S={stride}",
            ha="center", va="top",
            fontsize=9, color=C_NAVY,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.25")
        )

def save_training_history_csv(history_df, phase_df):
    history_df.to_csv(os.path.join(OUT_DIR, "training_history.csv"), index=False)
    phase_df.to_csv(os.path.join(OUT_DIR, "training_phase_summary.csv"), index=False)

def plot_training_convergence(history_df):
    epochs = history_df["epoch"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
    axes = axes.ravel()

    # Panel 1: total loss
    ax = axes[0]
    ax.plot(epochs, history_df["train_total"], color=C_TEAL, linewidth=2.2, label="Train total")
    ax.plot(epochs, history_df["val_total"], color=C_BLUE, linewidth=2.2, label="Val total")
    best_idx = history_df["val_total"].idxmin()
    ax.scatter(
        history_df.loc[best_idx, "epoch"],
        history_df.loc[best_idx, "val_total"],
        color=C_NAVY, s=50, zorder=3, label="Best val"
    )
    ax.set_title("Training convergence: total loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    add_curriculum_phase_shading(ax, CURRICULUM)

    # Panel 2: component train losses
    ax = axes[1]
    ax.plot(epochs, history_df["train_q"], color=C_BLUE, linewidth=2.0, label="q loss")
    ax.plot(epochs, history_df["train_w"], color=C_TEAL, linewidth=2.0, label="omega loss")
    ax.plot(epochs, history_df["train_unity"], color=C_BLUE_LIGHT, linewidth=2.0, label="unity loss")
    ax.plot(epochs, history_df["train_kin"], color=C_TEAL_LIGHT, linewidth=2.0, label="kinematic loss")
    ax.set_title("Training convergence: component losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, ncol=2)
    add_curriculum_phase_shading(ax, CURRICULUM)

    # Panel 3: validation metrics
    ax = axes[2]
    ax.plot(epochs, history_df["val_q"], color=C_BLUE, linewidth=2.0, label="Val q loss")
    ax.plot(epochs, history_df["val_w"], color=C_TEAL, linewidth=2.0, label="Val omega loss")
    ax.plot(epochs, history_df["val_kin"], color=C_TEAL_LIGHT, linewidth=2.0, label="Val kin loss")
    ax.set_title("Validation loss components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, ncol=2)
    add_curriculum_phase_shading(ax, CURRICULUM)

    # Panel 4: validation quaternion angle
    ax = axes[3]
    ax.plot(epochs, history_df["val_angle_deg"], color=C_NAVY, linewidth=2.2, label="Val q angle")
    ax.set_title("Validation quaternion angle error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Angle error (deg)")
    ax.legend(frameon=False)
    add_curriculum_phase_shading(ax, CURRICULUM)

    fig.suptitle("Fig. 1  Training convergence with curriculum phases", fontsize=14, color=C_NAVY)
    fig.savefig(os.path.join(OUT_DIR, "fig1_training_convergence.png"), dpi=300, bbox_inches="tight")
    plt.show()

def plot_error_metrics(predictions_q, targets_q, predictions_w, targets_w):
    # quaternion component errors
    quat_err_w = predictions_q[:, 0] - targets_q[:, 0]
    quat_err_x = predictions_q[:, 1] - targets_q[:, 1]
    quat_err_y = predictions_q[:, 2] - targets_q[:, 2]
    quat_err_z = predictions_q[:, 3] - targets_q[:, 3]
    quat_total_err = np.sqrt(np.sum((predictions_q - targets_q) ** 2, axis=1))

    # omega errors
    omega_err_x = predictions_w[:, 0] - targets_w[:, 0]
    omega_err_y = predictions_w[:, 1] - targets_w[:, 1]
    omega_err_z = predictions_w[:, 2] - targets_w[:, 2]
    omega_rmse_t = np.sqrt(np.mean((predictions_w - targets_w) ** 2, axis=1))

    # Record all error data to CSV
    error_df = pd.DataFrame({
        "time_step": np.arange(len(predictions_q)),
        "quat_err_w": quat_err_w,
        "quat_err_x": quat_err_x,
        "quat_err_y": quat_err_y,
        "quat_err_z": quat_err_z,
        "quat_total_err": quat_total_err,
        "omega_err_x": omega_err_x,
        "omega_err_y": omega_err_y,
        "omega_err_z": omega_err_z,
        "omega_rmse_t": omega_rmse_t,
    })
    error_df.to_csv(os.path.join(OUT_DIR, "error_metrics_for_plot.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)

    # Subplot 1: Quaternion errors (boxplot)
    ax = axes[0]
    bp = ax.boxplot(
        [quat_err_w, quat_err_x, quat_err_y, quat_err_z, quat_total_err],
        labels=["quat_w", "quat_x", "quat_y", "quat_z", "total quat error"],
        patch_artist=True,
        medianprops=dict(color=C_NAVY, linewidth=2)
    )
    box_colors = ["#138D90", "#FF6666", "#66FF66", "#6666FF", "#6C7A89"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_title("Quaternion component & total error (boxplot)")
    ax.set_ylabel("Error")
    ax.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2], bp["boxes"][3], bp["boxes"][4]],
              ["quat_w", "quat_x", "quat_y", "quat_z", "total quat error"],
              loc="upper right", frameon=True, fontsize=8, facecolor="white")

    # Subplot 2: Omega errors (boxplot)
    ax = axes[1]
    bp = ax.boxplot(
        [omega_err_x, omega_err_y, omega_err_z, omega_rmse_t],
        labels=["omega_x", "omega_y", "omega_z", "omega RMSE"],
        patch_artist=True,
        medianprops=dict(color=C_NAVY, linewidth=2)
    )
    box_colors = ["#FF6666", "#66FF66", "#6666FF", "#6C7A89"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_title("Omega component & RMSE error (boxplot)")
    ax.set_ylabel("Error")
    ax.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2], bp["boxes"][3]],
              ["omega_x", "omega_y", "omega_z", "omega RMSE"],
              loc="upper right", frameon=True, fontsize=8, facecolor="white")

    fig.suptitle("Error metrics", fontsize=14, color=C_NAVY)
    fig.savefig(os.path.join(OUT_DIR, "fig2_error_metrics.svg"), format="svg", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {os.path.join(OUT_DIR, 'fig2_error_metrics.svg')}")

def plot_prediction_examples(q, targets_q, predictions_q, omega, targets_w, predictions_w):
    quat_labels = ["w", "x", "y", "z"]
    omega_labels = ["wx", "wy", "wz"]
    t_steps_q = np.arange(len(targets_q))
    t_steps_w = np.arange(len(targets_w))

    # Quaternion example
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for i in range(4):
        axes[i].plot(t_steps_q, q[:, i], label="Input q", linewidth=1.8, color=C_TEAL_LIGHT)
        axes[i].plot(t_steps_q, targets_q[:, i], label="Target q_slow", linewidth=2.0, color=C_BLUE)
        axes[i].plot(t_steps_q, predictions_q[:, i], label="Predicted q_slow", linestyle="--", linewidth=2.0, color=C_TEAL)

        mae_i = np.mean(np.abs(predictions_q[:, i] - targets_q[:, i]))
        rmse_i = np.sqrt(np.mean((predictions_q[:, i] - targets_q[:, i]) ** 2))
        axes[i].set_title(f"q_{quat_labels[i]}  |  MAE={mae_i:.4e}, RMSE={rmse_i:.4e}")
        axes[i].set_ylabel("Value")
        axes[i].legend(frameon=False, fontsize=8)

    axes[2].set_xlabel("Time step")
    axes[3].set_xlabel("Time step")
    fig.suptitle("Fig. 3a  Model prediction example: quaternion", fontsize=14, color=C_NAVY)
    fig.savefig(os.path.join(OUT_DIR, "fig3a_prediction_quaternion.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Omega example
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True, constrained_layout=True)

    for i in range(3):
        axes[i].plot(t_steps_w, omega[:, i], label="Input omega", linewidth=1.8, color=C_TEAL_LIGHT)
        axes[i].plot(t_steps_w, targets_w[:, i], label="Target omega_slow", linewidth=2.0, color=C_BLUE)
        axes[i].plot(t_steps_w, predictions_w[:, i], label="Predicted omega_slow", linestyle="--", linewidth=2.0, color=C_TEAL)

        mae_i = np.mean(np.abs(predictions_w[:, i] - targets_w[:, i]))
        rmse_i = np.sqrt(np.mean((predictions_w[:, i] - targets_w[:, i]) ** 2))
        axes[i].set_title(f"{omega_labels[i]}  |  MAE={mae_i:.4e}, RMSE={rmse_i:.4e}")
        axes[i].set_ylabel("Value")
        axes[i].legend(frameon=False, fontsize=8)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Fig. 3b  Model prediction example: omega", fontsize=14, color=C_NAVY)
    fig.savefig(os.path.join(OUT_DIR, "fig3b_prediction_omega.png"), dpi=300, bbox_inches="tight")
    plt.show()

def save_prediction_csvs(q, targets_q, predictions_q, omega, targets_w, predictions_w):
    pred_q_t = torch.from_numpy(predictions_q).float()
    true_q_t = torch.from_numpy(targets_q).float()
    quat_angle_deg = quaternion_angle_error_deg_tensor(pred_q_t, true_q_t).cpu().numpy()

    omega_err = predictions_w - targets_w
    omega_abs_err = np.abs(omega_err)
    omega_rmse_t = np.sqrt(np.mean(omega_err**2, axis=1))

    # detailed time-series prediction data
    pred_df = pd.DataFrame({
        "time_step": np.arange(len(targets_q)),
        "q_input_w": q[:, 0], "q_input_x": q[:, 1], "q_input_y": q[:, 2], "q_input_z": q[:, 3],
        "q_target_w": targets_q[:, 0], "q_target_x": targets_q[:, 1], "q_target_y": targets_q[:, 2], "q_target_z": targets_q[:, 3],
        "q_pred_w": predictions_q[:, 0], "q_pred_x": predictions_q[:, 1], "q_pred_y": predictions_q[:, 2], "q_pred_z": predictions_q[:, 3],
        "omega_input_x": omega[:, 0], "omega_input_y": omega[:, 1], "omega_input_z": omega[:, 2],
        "omega_target_x": targets_w[:, 0], "omega_target_y": targets_w[:, 1], "omega_target_z": targets_w[:, 2],
        "omega_pred_x": predictions_w[:, 0], "omega_pred_y": predictions_w[:, 1], "omega_pred_z": predictions_w[:, 2],
    })
    pred_df.to_csv(os.path.join(OUT_DIR, "prediction_timeseries.csv"), index=False)

    # error time-series
    err_df = pd.DataFrame({
        "time_step": np.arange(len(targets_q)),
        "quat_angle_error_deg": quat_angle_deg,
        "omega_err_x": omega_err[:, 0],
        "omega_err_y": omega_err[:, 1],
        "omega_err_z": omega_err[:, 2],
        "omega_abs_err_x": omega_abs_err[:, 0],
        "omega_abs_err_y": omega_abs_err[:, 1],
        "omega_abs_err_z": omega_abs_err[:, 2],
        "omega_rmse_t": omega_rmse_t,
    })
    err_df.to_csv(os.path.join(OUT_DIR, "prediction_errors.csv"), index=False)

    # summary stats
    summary_rows = []
    quat_names = ["w", "x", "y", "z"]
    for i, name in enumerate(quat_names):
        err = predictions_q[:, i] - targets_q[:, i]
        summary_rows.append({
            "group": "quaternion",
            "component": f"q_{name}",
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mean_error": float(np.mean(err)),
            "std_error": float(np.std(err)),
        })

    omega_names = ["x", "y", "z"]
    for i, name in enumerate(omega_names):
        err = predictions_w[:, i] - targets_w[:, i]
        summary_rows.append({
            "group": "omega",
            "component": f"omega_{name}",
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mean_error": float(np.mean(err)),
            "std_error": float(np.std(err)),
        })

    summary_rows.append({
        "group": "trajectory",
        "component": "quaternion_angle_deg",
        "mae": float(np.mean(quat_angle_deg)),
        "rmse": float(np.sqrt(np.mean(quat_angle_deg**2))),
        "mean_error": float(np.mean(quat_angle_deg)),
        "std_error": float(np.std(quat_angle_deg)),
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "prediction_summary_statistics.csv"), index=False)

# -----------------------
# Training with:
#   Method 5: curriculum on sequence length
#   Method 6: overlapping windows
# -----------------------
best_val = float("inf")
best_state = None
best_epoch = -1
current_seg_len = None
current_stride = None
train_loader = None
val_loader = None
omega_mean = None
omega_std = None

history = []

for epoch in range(1, NUM_EPOCHS + 1):
    seg_len, stride = get_window_config_for_epoch(epoch)

    if seg_len != current_seg_len or stride != current_stride:
        current_seg_len = seg_len
        current_stride = stride
        print(
            f"\n=== Switching curriculum at epoch {epoch}: "
            f"seg_len={current_seg_len}, stride={current_stride} ==="
        )
        train_loader, val_loader, omega_mean, omega_std = make_loaders_for_window(current_seg_len, current_stride)

        omega_mean_t = torch.tensor(omega_mean.squeeze(), dtype=torch.float32, device=device).view(1, 1, 3)
        omega_std_t = torch.tensor(omega_std.squeeze(), dtype=torch.float32, device=device).view(1, 1, 3)

    model.train()

    running_total = 0.0
    running_q = 0.0
    running_w = 0.0
    running_unity = 0.0
    running_kin = 0.0
    batches = 0

    for xb, yqb, ywb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yqb = yqb.to(device, non_blocking=True)
        ywb = ywb.to(device, non_blocking=True)

        optimizer.zero_grad()

        q_pred_raw, q_pred, omega_pred_n, _ = model(xb)
        loss, loss_q, loss_w, loss_unity, loss_kin = compute_total_loss(
            q_pred_raw, q_pred, omega_pred_n, yqb, ywb, omega_mean_t, omega_std_t
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_total += loss.item()
        running_q += loss_q.item()
        running_w += loss_w.item()
        running_unity += loss_unity.item()
        running_kin += loss_kin.item()
        batches += 1

    train_total = running_total / max(batches, 1)
    train_q = running_q / max(batches, 1)
    train_w = running_w / max(batches, 1)
    train_unity = running_unity / max(batches, 1)
    train_kin = running_kin / max(batches, 1)

    val_stats = eval_full(val_loader, omega_mean_t, omega_std_t)

    if val_stats["total"] < best_val:
        best_val = val_stats["total"]
        best_epoch = epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    history.append({
        "epoch": epoch,
        "seg_len": current_seg_len,
        "stride": current_stride,
        "train_total": train_total,
        "train_q": train_q,
        "train_w": train_w,
        "train_unity": train_unity,
        "train_kin": train_kin,
        "val_total": val_stats["total"],
        "val_q": val_stats["q"],
        "val_w": val_stats["w"],
        "val_unity": val_stats["unity"],
        "val_kin": val_stats["kin"],
        "val_angle_deg": val_stats["angle_deg"],
        "is_best": int(epoch == best_epoch),
    })

    if epoch % 5 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d} | seg_len={current_seg_len} | stride={current_stride} | "
            f"train total: {train_total:.6f} | train q: {train_q:.6f} | "
            f"train w: {train_w:.6f} | train unity: {train_unity:.6f} | train kin: {train_kin:.6f} | "
            f"val total: {val_stats['total']:.6f} | val q angle (deg): {val_stats['angle_deg']:.4f}"
        )

model.load_state_dict(best_state)

print(f"\nBest val total loss: {best_val:.6f} at epoch {best_epoch}")

torch.save(best_state, os.path.join(OUT_DIR, "gru_qslow_omegaslow_best_curriculum_overlap_kin.pt"))
print("Model saved")

# -----------------------
# Save training history CSV
# -----------------------
history_df = pd.DataFrame(history)

phase_rows = []
for i, (start_ep, end_ep, seg_len, stride) in enumerate(CURRICULUM, start=1):
    mask = (history_df["epoch"] >= start_ep) & (history_df["epoch"] <= end_ep)
    phase_hist = history_df.loc[mask]
    phase_rows.append({
        "phase": i,
        "start_epoch": start_ep,
        "end_epoch": end_ep,
        "seg_len": seg_len,
        "stride": stride,
        "num_epochs": len(phase_hist),
        "mean_train_total": phase_hist["train_total"].mean(),
        "mean_val_total": phase_hist["val_total"].mean(),
        "mean_val_angle_deg": phase_hist["val_angle_deg"].mean(),
        "best_val_total_in_phase": phase_hist["val_total"].min(),
        "best_epoch_in_phase": phase_hist.loc[phase_hist["val_total"].idxmin(), "epoch"],
    })

phase_df = pd.DataFrame(phase_rows)
save_training_history_csv(history_df, phase_df)

# -----------------------
# Plot Fig.1: training convergence
# -----------------------
plot_training_convergence(history_df)

# -----------------------
# Step-by-step prediction
# -----------------------
print("\nStep-by-step prediction on segment_data_1228_1.csv")

q, q_slow, omega, omega_slow = load_file("segment_data_1228_1.csv")

inputs = np.concatenate([q, omega], axis=1).astype(np.float32)
targets_q = q_slow
targets_w = omega_slow

# reuse current omega normalization from final curriculum stage
X_norm = ((inputs - x_mean.squeeze()) / x_std.squeeze()).astype(np.float32)
omega_mean_final = omega_mean.squeeze()
omega_std_final = omega_std.squeeze()

# Save omega normalization stats for later use
np.save(os.path.join(OUT_DIR, "omega_normalization.npy"), np.stack([omega_mean_final, omega_std_final]))
model.eval()

predictions_q = []
predictions_w = []
hidden_state = None

with torch.no_grad():
    for t in range(len(X_norm)):
        x_t = torch.from_numpy(X_norm[t:t+1]).float().unsqueeze(0).to(device)  # (1, 1, 7)

        q_pred_raw, q_pred, omega_pred_n, hidden_state = model(x_t, hidden_state)

        if hidden_state is not None:
            hidden_state = hidden_state.detach()

        q_pred = q_pred.squeeze(0).cpu().numpy()   # (1, 4)
        q_pred = q_pred / (np.linalg.norm(q_pred, axis=1, keepdims=True) + 1e-8)
        q_pred = canonicalize_quaternion_array(q_pred)

        omega_pred_n = omega_pred_n.squeeze(0).cpu().numpy()  # (1, 3)
        omega_pred = omega_pred_n * omega_std_final + omega_mean_final

        predictions_q.append(q_pred[0])
        predictions_w.append(omega_pred[0])

predictions_q = np.array(predictions_q, dtype=np.float32)
predictions_w = np.array(predictions_w, dtype=np.float32)

# -----------------------
# Metrics
# -----------------------
pred_q_t = torch.from_numpy(predictions_q).float()
true_q_t = torch.from_numpy(targets_q).float()

mean_angle_deg = quaternion_angle_error_deg(pred_q_t, true_q_t)
quat_loss_eval = criterion_q(pred_q_t, true_q_t).item()
omega_mse_eval = np.mean((predictions_w - targets_w) ** 2)

with torch.no_grad():
    pred_q_t_b = pred_q_t.unsqueeze(0)
    pred_w_t_b = torch.from_numpy(predictions_w).float().unsqueeze(0)
    kin_eval = criterion_kin(pred_q_t_b, pred_w_t_b).item()

print("Mean quaternion angle error (deg):", mean_angle_deg)
print("Quaternion loss (q_pred vs q_slow):", quat_loss_eval)
print("Omega MSE (omega_pred vs omega_slow):", omega_mse_eval)
print("Kinematic loss on predicted trajectory:", kin_eval)

# -----------------------
# Save CSVs for prediction and error analysis
# -----------------------
save_prediction_csvs(q, targets_q, predictions_q, omega, targets_w, predictions_w)

# -----------------------
# Plot Fig.2: error metrics
# -----------------------
plot_error_metrics(predictions_q, targets_q, predictions_w, targets_w)

# -----------------------
# Plot Fig.3: model prediction examples
# -----------------------
plot_prediction_examples(q, targets_q, predictions_q, omega, targets_w, predictions_w)

print(f"\nAll outputs saved in: {OUT_DIR}")
print("Saved files:")
for fn in sorted(os.listdir(OUT_DIR)):
    print(" -", fn)
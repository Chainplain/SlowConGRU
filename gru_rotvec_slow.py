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

# -----------------------
# Data loading
# -----------------------
def load_file(path):
    df = pd.read_csv(path)

    q_array = df[["q_w", "q_x", "q_y", "q_z"]].to_numpy(dtype=np.float32)
    q_slow_array = df[["q_slow_w", "q_slow_x", "q_slow_y", "q_slow_z"]].to_numpy(dtype=np.float32)
    omega = df[["omega_x", "omega_y", "omega_z"]].to_numpy(dtype=np.float32)

    q_array = normalize_quaternion_array(q_array)
    q_slow_array = normalize_quaternion_array(q_slow_array)

    q_array = canonicalize_quaternion_array(q_array)
    q_slow_array = canonicalize_quaternion_array(q_slow_array)

    return q_array, q_slow_array, omega.astype(np.float32)

# Cache raw file data once
raw_data = {}
for path in DATA_FILES:
    print(f"Loading raw {path} ...")
    q, q_slow, omega = load_file(path)
    raw_data[path] = {
        "q": q,
        "q_slow": q_slow,
        "omega": omega,
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
def make_chunks(q, q_slow, omega, seg_len, stride):
    """
    Build overlapping windows.

    Inputs:
        q:       (N, 4)
        q_slow:  (N, 4)
        omega:   (N, 3)
        seg_len: window length
        stride:  hop length between consecutive windows

    Returns:
        X_chunks: list of (seg_len, 7)
        Y_chunks: list of (seg_len, 4)
    """
    assert seg_len > 0
    assert stride > 0

    inputs = np.concatenate([q, omega], axis=1).astype(np.float32)   # (N, 7)
    targets = q_slow.astype(np.float32)                              # (N, 4)

    N = len(inputs)
    X_chunks, Y_chunks = [], []

    if N < seg_len:
        return X_chunks, Y_chunks

    for s in range(0, N - seg_len + 1, stride):
        X_chunks.append(inputs[s:s + seg_len])
        Y_chunks.append(targets[s:s + seg_len])

    return X_chunks, Y_chunks

def build_chunked_dataset(seg_len, stride):
    all_X, all_Y = [], []

    for path in DATA_FILES:
        q = raw_data[path]["q"]
        q_slow = raw_data[path]["q_slow"]
        omega = raw_data[path]["omega"]

        X_chunks, Y_chunks = make_chunks(q, q_slow, omega, seg_len, stride)
        all_X.extend(X_chunks)
        all_Y.extend(Y_chunks)

    all_X = np.stack(all_X, axis=0).astype(np.float32)
    all_Y = np.stack(all_Y, axis=0).astype(np.float32)

    Xn = ((all_X - x_mean) / x_std).astype(np.float32)
    Yn = all_Y.copy()   # keep q_slow as raw unit quaternion

    return Xn, Yn

# -----------------------
# Dataset
# -----------------------
class QuaternionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def make_loaders_for_window(seg_len, stride, batch_size=BATCH_SIZE, val_ratio=VAL_RATIO):
    Xn, Yn = build_chunked_dataset(seg_len, stride)

    n_total = len(Xn)
    n_val = max(1, int(val_ratio * n_total))

    idx = np.random.permutation(n_total)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = QuaternionDataset(Xn[train_idx], Yn[train_idx])
    val_ds = QuaternionDataset(Xn[val_idx], Yn[val_idx])

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

    return train_loader, val_loader

# -----------------------
# GRU Model
# -----------------------
class GRUSlowQuaternionExtractor(nn.Module):
    """
    Predict q_slow from input [q, omega].

    Input:
        x: normalized [q(4), omega(3)]
    Output:
        q_pred: unit quaternion with q_pred[..., 0] >= 0
    """

    def __init__(self, input_mean, input_std, input_size=7, output_size=4,
                 hidden_size=64, num_layers=3, dropout=0.1):
        super().__init__()

        assert input_size == 7
        assert output_size == 4

        self.output_size = output_size
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

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
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

        delta = self.head(layer_input)  # (B, T, 4)

        q_in_norm = x[:, :, :4]
        q_mean = self.input_mean[:4].view(1, 1, 4)
        q_std = self.input_std[:4].view(1, 1, 4)

        q_in = q_in_norm * q_std + q_mean
        q_in = q_in / (torch.norm(q_in, dim=-1, keepdim=True) + 1e-8)
        q_in = canonicalize_quaternion_tensor(q_in)

        q_pred = q_in + delta
        q_pred = q_pred / (torch.norm(q_pred, dim=-1, keepdim=True) + 1e-8)
        q_pred = canonicalize_quaternion_tensor(q_pred)

        return q_pred, hT

# -----------------------
# Quaternion loss
# -----------------------
class QuaternionGeodesicLoss(nn.Module):
    """
    Loss is computed strictly between q_pred and q_slow.

    loss = mean(1 - <q_pred, q_slow>^2)
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, q_pred, q_slow):
        q_pred = q_pred / (torch.norm(q_pred, dim=-1, keepdim=True) + self.eps)
        q_slow = q_slow / (torch.norm(q_slow, dim=-1, keepdim=True) + self.eps)

        dot = torch.sum(q_pred * q_slow, dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)

        loss = 1.0 - dot * dot
        return loss.mean()

# -----------------------
# Model / optimizer
# -----------------------
model = GRUSlowQuaternionExtractor(
    input_mean=x_mean,
    input_std=x_std,
    input_size=7,
    output_size=4,
    hidden_size=64,
    num_layers=3,
    dropout=0.1
).to(device)

criterion = QuaternionGeodesicLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# -----------------------
# Evaluation helpers
# -----------------------
def quaternion_angle_error_deg(q_pred, q_true):
    q_pred = q_pred / (torch.norm(q_pred, dim=-1, keepdim=True) + 1e-8)
    q_true = q_true / (torch.norm(q_true, dim=-1, keepdim=True) + 1e-8)

    dot = torch.sum(q_pred * q_true, dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)

    angle = 2.0 * torch.acos(dot)
    angle_deg = angle * 180.0 / np.pi
    return angle_deg.mean().item()

def eval_loss(loader):
    model.eval()
    total = 0.0
    n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            q_pred, _ = model(xb)
            loss = criterion(q_pred, yb)

            total += loss.item() * xb.size(0)
            n += xb.size(0)

    return total / max(n, 1)

def eval_angle_deg(loader):
    model.eval()
    total = 0.0
    n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            q_pred, _ = model(xb)
            ang = quaternion_angle_error_deg(q_pred, yb)

            total += ang * xb.size(0)
            n += xb.size(0)

    return total / max(n, 1)

def get_window_config_for_epoch(epoch):
    for start_ep, end_ep, seg_len, stride in CURRICULUM:
        if start_ep <= epoch <= end_ep:
            return seg_len, stride
    _, _, seg_len, stride = CURRICULUM[-1]
    return seg_len, stride

# -----------------------
# Training with:
#    curriculum on sequence length
#    overlapping windows
# -----------------------
best_val = float("inf")
best_state = None
current_seg_len = None
current_stride = None
train_loader = None
val_loader = None

for epoch in range(1, NUM_EPOCHS + 1):
    seg_len, stride = get_window_config_for_epoch(epoch)

    # Rebuild loaders only when curriculum stage changes
    if seg_len != current_seg_len or stride != current_stride:
        current_seg_len = seg_len
        current_stride = stride
        print(
            f"\n=== Switching curriculum at epoch {epoch}: "
            f"seg_len={current_seg_len}, stride={current_stride} ==="
        )
        train_loader, val_loader = make_loaders_for_window(current_seg_len, current_stride)

    model.train()

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()

        q_pred, _ = model(xb)
        loss = criterion(q_pred, yb)   # strictly q_pred vs q_slow

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    val = eval_loss(val_loader)

    if val < best_val:
        best_val = val
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 5 == 0 or epoch == 1:
        val_ang = eval_angle_deg(val_loader)
        print(
            f"Epoch {epoch:03d} | seg_len={current_seg_len} | stride={current_stride} | "
            f"val quat loss: {val:.6f} | val angle err (deg): {val_ang:.4f}"
        )

model.load_state_dict(best_state)

print(f"\nBest val quaternion loss: {best_val:.6f}")

torch.save(best_state, "gru_qslow_best_curriculum_overlap.pt")
print("Model saved")

# -----------------------
# Step-by-step prediction
# -----------------------
print("\nStep-by-step prediction on segment_data_1228_1.csv")

q, q_slow, omega = load_file("segment_data_1228_1.csv")

inputs = np.concatenate([q, omega], axis=1).astype(np.float32)
targets = q_slow

X_norm = ((inputs - x_mean.squeeze()) / x_std.squeeze()).astype(np.float32)
Y_true = targets

model.eval()

predictions = []
hidden_state = None

with torch.no_grad():
    for t in range(len(X_norm)):
        x_t = torch.from_numpy(X_norm[t:t+1]).float().unsqueeze(0).to(device)  # (1, 1, 7)

        q_pred, hidden_state = model(x_t, hidden_state)

        if hidden_state is not None:
            hidden_state = hidden_state.detach()

        q_pred = q_pred.squeeze(0).cpu().numpy()   # (1, 4)
        q_pred = q_pred / (np.linalg.norm(q_pred, axis=1, keepdims=True) + 1e-8)
        q_pred = canonicalize_quaternion_array(q_pred)

        predictions.append(q_pred[0])

predictions = np.array(predictions, dtype=np.float32)

# -----------------------
# Plot
# -----------------------
quat_labels = ["w", "x", "y", "z"]
t_steps = np.arange(len(Y_true))

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

for i in range(4):
    axes[i].plot(t_steps, q[:, i], label="Input q", linewidth=2, color="#85c0ea")
    axes[i].plot(t_steps, Y_true[:, i], label="Target q_slow", linewidth=2, color="#336ad8")
    axes[i].plot(t_steps, predictions[:, i], label="Predicted q_slow", linestyle="--", color="#de2040", zorder=3)

    axes[i].set_ylabel(quat_labels[i])
    axes[i].legend()
    axes[i].grid(True)

axes[-1].set_xlabel("Time step")
axes[0].set_title("GRU step-by-step prediction of q_slow (curriculum + overlap)")

plt.tight_layout()
plt.show()

# -----------------------
# Metrics
# -----------------------
pred_t = torch.from_numpy(predictions).float()
true_t = torch.from_numpy(Y_true).float()

mean_angle_deg = quaternion_angle_error_deg(pred_t, true_t)
quat_loss_eval = criterion(pred_t, true_t).item()

print("Mean quaternion angle error (deg):", mean_angle_deg)
print("Quaternion loss (q_pred vs q_slow):", quat_loss_eval)
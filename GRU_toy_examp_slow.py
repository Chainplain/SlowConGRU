import numpy as np
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

# -----------------------
# Toy signal generator
# x(t) = slow(t) + fast(t) (+ noise)
# target y(t) = slow(t)
# -----------------------
def make_mixture(
    T: int,
    dt: float,
    slow_freq: float,
    fast_freq: float,
    slow_amp: float = 1.0,
    fast_amp: float = 0.4,
    slow_phase: float = 0.0,
    fast_phase: float = 0.0,
    noise_std: float = 0.02,
):
    t = np.arange(T) * dt
    slow = slow_amp * np.sin(2 * np.pi * slow_freq * t + slow_phase)
    fast = fast_amp * np.sin(2 * np.pi * fast_freq * t + fast_phase)
    x = slow + fast + noise_std * np.random.randn(T)
    return t, x, slow, fast

# -----------------------
# Dataset: many random sequences, train GRU to recover slow part
# -----------------------
class MixtureDataset(Dataset):
    def __init__(
        self,
        n_seq: int,
        T: int,
        dt: float,
        slow_freq_range=(0.02, 0.06),
        fast_freq_range=(0.25, 0.6),
        slow_amp_range=(0.7, 1.3),
        fast_amp_range=(0.2, 0.7),
        noise_std=0.02,
    ):
        self.X = []
        self.Y = []
        for _ in range(n_seq):
            slow_f = np.random.uniform(*slow_freq_range)
            fast_f = np.random.uniform(*fast_freq_range)
            slow_a = np.random.uniform(*slow_amp_range)
            fast_a = np.random.uniform(*fast_amp_range)
            slow_p = np.random.uniform(0, 2*np.pi)
            fast_p = np.random.uniform(0, 2*np.pi)

            _, x, slow, _ = make_mixture(
                T=T,
                dt=dt,
                slow_freq=slow_f,
                fast_freq=fast_f,
                slow_amp=slow_a,
                fast_amp=fast_a,
                slow_phase=slow_p,
                fast_phase=fast_p,
                noise_std=noise_std,
            )
            # Shape: (T, 1)
            self.X.append(x.astype(np.float32).reshape(T, 1))
            self.Y.append(slow.astype(np.float32).reshape(T, 1))

        self.X = np.stack(self.X, axis=0)  # (N, T, 1)
        self.Y = np.stack(self.Y, axis=0)  # (N, T, 1)

        # Normalize using training-set stats (per feature)
        x_mean = self.X.mean(axis=(0, 1), keepdims=True)
        x_std = self.X.std(axis=(0, 1), keepdims=True) + 1e-6
        y_mean = self.Y.mean(axis=(0, 1), keepdims=True)
        y_std = self.Y.std(axis=(0, 1), keepdims=True) + 1e-6

        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

        self.Xn = (self.X - x_mean) / x_std
        self.Yn = (self.Y - y_mean) / y_std

    def __len__(self):
        return self.Xn.shape[0]

    def __getitem__(self, idx):
        # Return torch tensors shaped (T, 1)
        return torch.from_numpy(self.Xn[idx]), torch.from_numpy(self.Yn[idx])

# -----------------------
# GRU model: sequence-to-sequence regression
# -----------------------
class GRUSlowExtractor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,  # (B, T, C)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Helpful init for RNN stability
        for name, p in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x, h0=None):
        # x: (B, T, 1)
        z, hT = self.gru(x, h0)   # z: (B, T, H)
        y = self.head(z)          # y: (B, T, 1)
        return y, hT

# -----------------------
# Train
# -----------------------
T = 400
dt = 0.1

train_ds = MixtureDataset(n_seq=2000, T=T, dt=dt, noise_std=0.02)
val_ds   = MixtureDataset(n_seq=200,  T=T, dt=dt, noise_std=0.02)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

model = GRUSlowExtractor(hidden_size=64, num_layers=2, dropout=0.0).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def eval_loss(loader):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            total += loss.item() * xb.size(0)
            n += xb.size(0)
    return total / max(n, 1)

best_val = float("inf")
best_state = None

for epoch in range(1, 31):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(device)  # (B,T,1)
        yb = yb.to(device)  # (B,T,1)

        optimizer.zero_grad()
        pred, _ = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    val = eval_loss(val_loader)
    if val < best_val:
        best_val = val
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | val MSE (normalized): {val:.6f}")

# restore best
if best_state is not None:
    model.load_state_dict(best_state)

# -----------------------
# Test on a fresh sequence with fixed frequencies to visualize
# -----------------------
t, x_raw, slow_raw, fast_raw = make_mixture(
    T=T,
    dt=dt,
    slow_freq=0.04,
    fast_freq=0.45,
    slow_amp=1.0,
    fast_amp=0.5,
    slow_phase=0.7,
    fast_phase=1.3,
    noise_std=0.02,
)

# Create fast component with gradually changing frequency
t_arr = np.arange(T) * dt
fast_raw = np.zeros(T)
for i in range(T):
    freq = 0.25 + (0.6 - 0.4) * (i / T)  # linearly vary from 0.25 to 0.65
    fast_raw[i] = 0.5 * np.sin(2 * np.pi * freq * t_arr[i] + 1.3)

x_raw = slow_raw + fast_raw + 0.02 * np.random.randn(T)

# normalize with TRAIN stats
x_n = ((x_raw.reshape(T,1) - train_ds.x_mean.squeeze()) / train_ds.x_std.squeeze()).astype(np.float32)
x_torch = torch.from_numpy(x_n).unsqueeze(0).to(device)  # (1,T,1)

model.eval()
with torch.no_grad():
    yhat_n_list = []
    h = None
    for t_step in range(x_torch.size(1)):
        x_t = x_torch[:, t_step:t_step+1, :]  # (1, 1, 1)
        y_t, h = model(x_t, h)  # (1, 1, 1)
        yhat_n_list.append(y_t)
    yhat_n = torch.cat(yhat_n_list, dim=1)  # (1, T, 1)
    yhat_n = yhat_n.squeeze(0).cpu().numpy().reshape(T,1)

# denormalize to original slow scale
yhat = yhat_n * train_ds.y_std.squeeze() + train_ds.y_mean.squeeze()
yhat = yhat.reshape(T)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(12, 4))
plt.plot(t, x_raw, label="Input x(t) = slow + fast + noise", alpha=0.6)
plt.plot(t, slow_raw, label="True slow(t)", linewidth=2)
plt.plot(t, yhat, label="GRU predicted slow(t)", linewidth=2, linestyle="--")
plt.plot(t, fast_raw, label="Fast(t) (for reference)", alpha=0.35)
plt.legend()
plt.title("GRU learns mapping: mixed signal -> slow component")
plt.tight_layout()

plt.savefig('gru_slow_extractor_result.svg', dpi=150, bbox_inches='tight')
plt.show()
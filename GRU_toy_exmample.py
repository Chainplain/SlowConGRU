import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Generate Train Data
# -----------------------
T = 1000
t = np.linspace(0, 100, T)

sin_wave = np.sin(t)
cos_wave = np.cos(t)

# Train tensors: (T, B, 1) where B=1
x = torch.tensor(sin_wave, dtype=torch.float32).view(T, 1, 1)
y = torch.tensor(cos_wave, dtype=torch.float32).view(T, 1, 1)

# -----------------------
# Generate Test Data (phase delayed by pi/2)
# -----------------------
# "Delayed pi/2 phase" means shift the input signal forward by +pi/2 in phase:
# sin(t + pi/2) = cos(t)
# cos(t + pi/2) = -sin(t)
t_test = t.copy()
x_test_wave = np.sin(t_test + np.pi / 2)  # = cos(t)
y_test_wave = np.cos(t_test + np.pi / 2)  # = -sin(t)

x_test = torch.tensor(x_test_wave, dtype=torch.float32).view(T, 1, 1)
y_test = torch.tensor(y_test_wave, dtype=torch.float32).view(T, 1, 1)

# -----------------------
# Define GRU Model (with streaming step)
# -----------------------
class GRUModel(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None):
        """
        Full-sequence forward.
        x: (T, B, 1)
        h0: (1, B, H) or None
        returns:
          out: (T, B, 1)
          hT: (1, B, H)
        """
        out, hT = self.gru(x, h0)
        out = self.fc(out)
        return out, hT

    def step(self, x_t, h=None):
        """
        One-step (streaming) forward.
        x_t: (B, 1) or (B,) or scalar
        h:   (1, B, H) or None
        returns:
          y_t: (B, 1)
          h:   (1, B, H)
        """
        if x_t.dim() == 0:
            x_t = x_t.view(1, 1)        # scalar -> (1,1)
        elif x_t.dim() == 1:
            x_t = x_t.unsqueeze(-1)     # (B,) -> (B,1)

        x_t = x_t.unsqueeze(0)          # (1, B, 1)
        out, h = self.gru(x_t, h)       # (1, B, H)
        y_t = self.fc(out)              # (1, B, 1)
        return y_t.squeeze(0), h        # (B, 1), (1, B, H)

# -----------------------
# Setup
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GRUModel(hidden_size=64).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x = x.to(device)
y = y.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

# -----------------------
# Training Loop (streaming-style)
# -----------------------
for epoch in range(300):
    model.train()
    optimizer.zero_grad()

    h = None
    loss = 0.0

    for i in range(T):
        y_pred_i, h = model.step(x[i, 0], h)
        loss = loss + criterion(y_pred_i.view(1, 1, 1), y[i])

    loss = loss / T
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}")

# -----------------------
# Evaluation on Test (full-sequence and streaming)
# -----------------------
model.eval()

with torch.no_grad():
    # Full-sequence test
    yhat_test_full, _ = model(x_test)
    test_loss_full = criterion(yhat_test_full, y_test).item()
    yhat_test_full_np = yhat_test_full.squeeze().cpu().numpy()

    # Streaming test
    h = None
    yhat_test_stream = []
    for i in range(T):
        y_t, h = model.step(x_test[i, 0], h)
        yhat_test_stream.append(y_t.item())
    yhat_test_stream = np.array(yhat_test_stream)

    test_loss_stream = np.mean((yhat_test_stream - y_test.squeeze().cpu().numpy()) ** 2)

print(f"Test Loss (full sequence): {test_loss_full:.6f}")
print(f"Test Loss (streaming):     {test_loss_stream:.6f}")

# -----------------------
# Plot Results
# -----------------------
plt.figure(figsize=(11, 4))

plt.plot(t_test, y_test_wave, label="True y_test = cos(t + pi/2) = -sin(t)")
plt.plot(t_test, yhat_test_full_np, label="Pred (test, full sequence)", linestyle="--")
plt.plot(t_test, yhat_test_stream, label="Pred (test, streaming)", alpha=0.85)

plt.title("GRU trained on (sin(t) -> cos(t)), tested on pi/2 phase-shifted signals")
plt.legend()
plt.tight_layout()
plt.show()
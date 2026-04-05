import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def canonicalize_quaternion_array(q):
    q = q.copy()
    flip_mask = q[:, 0] < 0.0
    q[flip_mask] *= -1.0
    return q


def canonicalize_quaternion_tensor(q):
    sign = torch.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q * sign


class GRUSlowStateExtractor(nn.Module):
    def __init__(
        self,
        input_mean,
        input_std,
        input_size=7,
        q_output_size=4,
        w_output_size=3,
        hidden_size=64,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()

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
                    batch_first=True,
                )
            )

            if in_size == hidden_size:
                self.residual_projs.append(nn.Identity())
            else:
                self.residual_projs.append(nn.Linear(in_size, hidden_size))

        self.q_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, q_output_size),
        )

        self.w_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, w_output_size),
        )

    def forward(self, x, h0=None):
        layer_input = x
        hT_list = []

        for layer_idx, (gru_layer, res_proj) in enumerate(zip(self.layers, self.residual_projs)):
            h_init = None
            if h0 is not None:
                h_init = h0[layer_idx : layer_idx + 1]

            z, h_layer = gru_layer(layer_input, h_init)
            hT_list.append(h_layer)

            residual = res_proj(layer_input)
            z = z + residual

            if layer_idx < self.num_layers - 1:
                z = self.dropout_layer(z)

            layer_input = z

        hT = torch.cat(hT_list, dim=0) if hT_list else None

        delta_q = self.q_head(layer_input)
        omega_pred_n = self.w_head(layer_input)

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


def load_model(checkpoint_path, device):
    model = GRUSlowStateExtractor(
        input_mean=np.zeros((1, 1, 7), dtype=np.float32),
        input_std=np.ones((1, 1, 7), dtype=np.float32),
        input_size=7,
        q_output_size=4,
        w_output_size=3,
        hidden_size=64,
        num_layers=3,
        dropout=0.1,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_omega_normalization(omega_norm_path):
    data = np.load(omega_norm_path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.dtype == object:
        maybe_obj = data.item()
        if isinstance(maybe_obj, dict):
            if "omega_mean" in maybe_obj and "omega_std" in maybe_obj:
                return np.asarray(maybe_obj["omega_mean"], dtype=np.float32), np.asarray(maybe_obj["omega_std"], dtype=np.float32)
            if "mean" in maybe_obj and "std" in maybe_obj:
                return np.asarray(maybe_obj["mean"], dtype=np.float32), np.asarray(maybe_obj["std"], dtype=np.float32)

    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 2 and data.shape[0] == 2:
        return data[0], data[1]

    if data.ndim == 1 and data.shape[0] == 6:
        return data[:3], data[3:]

    raise ValueError(f"Unsupported omega normalization shape: {data.shape}")


def load_input(csv_path):
    df = pd.read_csv(csv_path)
    q = df[["q_w", "q_x", "q_y", "q_z"]].to_numpy(dtype=np.float32)
    omega = df[["omega_x", "omega_y", "omega_z"]].to_numpy(dtype=np.float32)

    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    q = canonicalize_quaternion_array(q)

    return q, omega


def run_step_by_step_prediction(model, q, omega, omega_mean_final, omega_std_final, device):
    inputs = np.concatenate([q, omega], axis=1).astype(np.float32)

    x_mean = model.input_mean.detach().cpu().numpy().astype(np.float32)
    x_std = model.input_std.detach().cpu().numpy().astype(np.float32)

    X_norm = ((inputs - x_mean) / x_std).astype(np.float32)

    predictions_q = []
    predictions_w = []
    hidden_state = None

    with torch.no_grad():
        for t in range(len(X_norm)):
            x_t = torch.from_numpy(X_norm[t : t + 1]).float().unsqueeze(0).to(device)  # (1, 1, 7)

            q_pred_raw, q_pred, omega_pred_n, hidden_state = model(x_t, hidden_state)

            if hidden_state is not None:
                hidden_state = hidden_state.detach()

            q_pred = q_pred.squeeze(0).cpu().numpy()  # (1, 4)
            q_pred = q_pred / (np.linalg.norm(q_pred, axis=1, keepdims=True) + 1e-8)
            q_pred = canonicalize_quaternion_array(q_pred)

            omega_pred_n = omega_pred_n.squeeze(0).cpu().numpy()  # (1, 3)
            omega_pred = omega_pred_n * omega_std_final + omega_mean_final

            predictions_q.append(q_pred[0])
            predictions_w.append(omega_pred[0])

    predictions_q = np.array(predictions_q, dtype=np.float32)
    predictions_w = np.array(predictions_w, dtype=np.float32)

    return predictions_q, predictions_w


def plot_prediction_vs_input(q, omega, predictions_q, predictions_w, output_path):
    t = np.arange(len(q))
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True, constrained_layout=True)

    q_labels = ["w", "x", "y", "z"]
    for i in range(4):
        ax = axes[i, 0]
        ax.plot(t, q[:, i], color="#7AA6D1", linewidth=1.8, label="Input q")
        ax.plot(t, predictions_q[:, i], color="#138D90", linestyle="--", linewidth=2.0, label="Pred q_slow")
        ax.set_ylabel(f"q_{q_labels[i]}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    w_labels = ["x", "y", "z"]
    for i in range(3):
        ax = axes[i, 1]
        ax.plot(t, omega[:, i], color="#B5D99C", linewidth=1.8, label="Input omega")
        ax.plot(t, predictions_w[:, i], color="#1F4E79", linestyle="--", linewidth=2.0, label="Pred omega_slow")
        ax.set_ylabel(f"omega_{w_labels[i]}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    axes[3, 1].axis("off")

    axes[3, 0].set_xlabel("Time step")
    axes[2, 1].set_xlabel("Time step")

    axes[0, 0].set_title("Quaternion: prediction vs input")
    axes[0, 1].set_title("Omega: prediction vs input")

    fig.suptitle("Loaded GRU model predictions", fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Load trained GRU and generate prediction plot.")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("training_figures_and_csv", "gru_qslow_omegaslow_best_curriculum_overlap_kin.pt"),
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--omega-norm",
        dest="omega_norm",
        default=os.path.join("training_figures_and_csv", "omega_normalization.npy"),
        help="Path to omega_normalization.npy",
    )
    parser.add_argument(
        "--input-csv",
        dest="input_csv",
        default="segment_data_1228_1.csv",
        help="CSV file used for inference",
    )
    parser.add_argument(
        "--output-plot",
        dest="output_plot",
        default=os.path.join("training_figures_and_csv", "load_gru_prediction_vs_input.png"),
        help="Where to save plot image",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.omega_norm):
        raise FileNotFoundError(f"Omega normalization file not found: {args.omega_norm}")
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    os.makedirs(os.path.dirname(args.output_plot) or ".", exist_ok=True)

    model = load_model(args.checkpoint, device)
    omega_mean_final, omega_std_final = load_omega_normalization(args.omega_norm)
    q, omega = load_input(args.input_csv)

    predictions_q, predictions_w = run_step_by_step_prediction(
        model=model,
        q=q,
        omega=omega,
        omega_mean_final=omega_mean_final,
        omega_std_final=omega_std_final,
        device=device,
    )

    plot_prediction_vs_input(q, omega, predictions_q, predictions_w, args.output_plot)

    print(f"Saved prediction plot: {args.output_plot}")
    print(f"Predicted q shape: {predictions_q.shape}")
    print(f"Predicted omega shape: {predictions_w.shape}")


if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Plot-only script
# Loads saved CSV files and re-generates figures
# without re-training the model.
# =========================================================

# -----------------------
# Config
# -----------------------
OUT_DIR = "training_figures_and_csv"
FIG_DIR = os.path.join(OUT_DIR, "replotted_figures")
os.makedirs(FIG_DIR, exist_ok=True)

FIGSIZE = (10, 6)

# Curriculum: (start_epoch, end_epoch, seg_len, stride)
CURRICULUM = [
    (1, 20, 50, 10),
    (21, 50, 100, 20),
    (51, 100, 200, 50),
]

# -----------------------
# Theme colors (teal / deep blue family)
# -----------------------
C_TEAL = "#138D90"
C_TEAL_LIGHT = "#7FD3D4"
C_BLUE = "#1F4E79"
C_BLUE_LIGHT = "#7AA6D1"
C_NAVY = "#0B2545"
C_GRID = "#D9E6F2"
PHASE_COLORS = ["#D8F0F0", "#DCE8F8", "#EAF2FB"]

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35
plt.rcParams["grid.color"] = C_GRID


# -----------------------
# Helpers
# -----------------------
def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Please make sure the CSV file exists in {OUT_DIR}."
        )


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


def plot_training_convergence(history_df):
    epochs = history_df["epoch"].to_numpy()

    # Distinctive colors for each loss
    color_q = "#E24A33"      # orange-red
    color_w = "#348ABD"      # blue
    color_unity = "#988ED5"  # purple
    color_kin = "#8EBA42"    # green
    color_val = "#FFB547"    # yellow-orange
    color_train_total = "#222222"  # black for train total

    fig, axes = plt.subplots(1, 1, figsize=FIGSIZE, constrained_layout=True)
    ax = axes
    ax.plot(epochs, history_df["train_q"], color=color_q, linewidth=2.0, label="Train q loss")
    ax.plot(epochs, history_df["train_w"], color=color_w, linewidth=2.0, label="Train omega loss")
    ax.plot(epochs, history_df["train_unity"], color=color_unity, linewidth=2.0, label="Train unity loss")
    ax.plot(epochs, history_df["train_kin"], color=color_kin, linewidth=2.0, label="Train kinematic loss")
    ax.plot(epochs, history_df["train_total"], color=color_train_total, linewidth=2.2, label="Train total loss")
    ax.plot(epochs, history_df["val_total"], color=color_val, linewidth=2.2, label="Val total loss")
    ax.set_title("Training convergence: individual losses (log scale)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=True, fontsize=12, facecolor="white")
    ax.tick_params(axis='both', labelsize=12)
    add_curriculum_phase_shading(ax, CURRICULUM)

    fig.suptitle("Training convergence with curriculum phases", fontsize=14, color=C_NAVY)
    save_path = os.path.join(FIG_DIR, "fig1_training_convergence.svg")
    fig.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_error_metrics(errors_df):
    error_csv_path = os.path.join(OUT_DIR, "error_metrics_for_plot.csv")
    if os.path.exists(error_csv_path):
        error_df = pd.read_csv(error_csv_path)
        quat_total_err = error_df["quat_total_err"].to_numpy()
        omega_err_x = error_df["omega_err_x"].to_numpy()
        omega_err_y = error_df["omega_err_y"].to_numpy()
        omega_err_z = error_df["omega_err_z"].to_numpy()
        omega_rmse_t = error_df["omega_rmse_t"].to_numpy()
    else:
        quat_total_err = errors_df["quat_total_err"].to_numpy()
        omega_err_x = errors_df["omega_err_x"].to_numpy()
        omega_err_y = errors_df["omega_err_y"].to_numpy()
        omega_err_z = errors_df["omega_err_z"].to_numpy()
        omega_rmse_t = errors_df["omega_rmse_t"].to_numpy()

    # Remove total quaternion error values larger than 0.25
    filtered_quat_total_err = quat_total_err[quat_total_err <= 0.25]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)

    # Subplot 1: Total quaternion error (boxplot)
    ax = axes[0]
    bp = ax.boxplot(
        [filtered_quat_total_err],
        labels=["total quat error"],
        patch_artist=True,
        medianprops=dict(color=C_NAVY, linewidth=2)
    )
    box_colors = ["#6C7A89"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_title("Total quaternion error (boxplot)")
    ax.set_ylabel("Error", fontsize=12)
    ax.legend([bp["boxes"][0]], ["total quat error"], loc="upper right", frameon=True, fontsize=12, facecolor="white")
    ax.tick_params(axis='both', labelsize=12)

    # Subplot 2: Omega errors (boxplot)
    ax = axes[1]
    bp = ax.boxplot(
        [omega_err_x, omega_err_y, omega_err_z, omega_rmse_t],
        labels=["omega_x", "omega_y", "omega_z", "omega RMSE"],
        patch_artist=True,
        medianprops=dict(color=C_NAVY, linewidth=2)
    )
    box_colors = ['#FF6666', "#9FE3C7", "#819CF2", "#6C7A89"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_title("Omega component & RMSE error (boxplot)")
    ax.set_ylabel("Error", fontsize=12)
    ax.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2], bp["boxes"][3]],
              [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$", r"$\omega_{RMSE}$"],
              loc="upper right", frameon=True, fontsize=12, facecolor="white")
    ax.tick_params(axis='both', labelsize=12)

    # fig.suptitle(fontsize=14, color=C_NAVY)
    save_path = os.path.join(FIG_DIR, "fig2_error_metrics.svg")
    fig.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_prediction_examples(pred_df):
    quat_labels = ["w", "x", "y", "z"]
    omega_labels = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
    t = pred_df["time_step"].to_numpy()

    # Convert time_step to seconds, assuming 180 Hz sampling
    dt = 1.0 / 180.0
    time = t * dt
    # Select time window from 40s to 70s
    mask = (time >= 40) & (time <= 70)
    time = time[mask]

    # quaternion arrays
    q_input = pred_df[["q_input_w", "q_input_x", "q_input_y", "q_input_z"]].to_numpy()[mask]
    q_target = pred_df[["q_target_w", "q_target_x", "q_target_y", "q_target_z"]].to_numpy()[mask]
    q_pred = pred_df[["q_pred_w", "q_pred_x", "q_pred_y", "q_pred_z"]].to_numpy()[mask]

    # omega arrays
    w_input = pred_df[["omega_input_x", "omega_input_y", "omega_input_z"]].to_numpy()[mask]
    w_target = pred_df[["omega_target_x", "omega_target_y", "omega_target_z"]].to_numpy()[mask]
    w_pred = pred_df[["omega_pred_x", "omega_pred_y", "omega_pred_z"]].to_numpy()[mask]

    # Update quaternion figure to use corresponding hues with deeper variations
    quat_colors = ['teal', 'red', 'green', 'blue']
    quat_input_colors = ['#4CB3B4', '#FF6666', "#9FE3C7", "#819CF2"]  # Deeper variations
    quat_target_colors = ['#138D90', '#990000', '#006600', '#000099']  # Darker variations
    # Adjust quaternion figure predicted linewidth and color saturation
    quat_pred_color = '#FF66CC'  # Less saturated magenta

    fig, axes = plt.subplots(4, 1, figsize=FIGSIZE, sharex=True, constrained_layout=True)

    for i in range(4):
        axes[i].plot(time, q_input[:, i], label="Input  $q$", linewidth=1.8, color=quat_input_colors[i])
        axes[i].plot(time, q_target[:, i], label="Target $q_{slow}$", linewidth=2.0, color=quat_target_colors[i])
        axes[i].plot(time, q_pred[:, i], label="Predicted $q_{slow}$", linestyle="--", linewidth=1, color=quat_pred_color, zorder=3)  # Reduced linewidth

        err = q_pred[:, i] - q_target[:, i]
        mae_i = np.mean(np.abs(err))
        rmse_i = np.sqrt(np.mean(err ** 2))

        axes[i].set_title(f"$q_{quat_labels[i]}$  |  MAE={mae_i:.4e}, RMSE={rmse_i:.4e}")
        axes[i].set_ylabel(f"$q_{quat_labels[i]}$", fontsize=14)
        axes[i].legend(loc="upper right", frameon=True, fontsize=14, facecolor="white")
        axes[i].tick_params(axis='both', labelsize=12)
    axes[-1].set_xlabel("Time (s)", fontsize=14)
    fig.suptitle("Model prediction example: quaternion", fontsize=14, color=C_NAVY)
    save_path = os.path.join(FIG_DIR, "fig3a_prediction_quaternion.svg")
    fig.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Update omega figure to use corresponding hues with variations
    omega_colors = ['#990000', '#006600', '#000099']  # Darker variations
    omega_input_colors = ['#FFCCCC', '#CCFFCC', '#CCCCFF']  # Softer variations
    # Adjust omega figure predicted linewidth and color saturation
    omega_pred_color = '#FF66CC'  # Less saturated magenta

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True, constrained_layout=True)

    for i in range(3):
        axes[i].plot(time, w_input[:, i], label="Input $\omega_{slow}$", linewidth=1.8, color=omega_input_colors[i])
        axes[i].plot(time, w_target[:, i], label="Target  $\omega_{slow}$", linewidth=2.0, color=omega_colors[i])
        axes[i].plot(time, w_pred[:, i], label="Predicted $\omega_{slow}$", linestyle="--", linewidth=1, color=omega_pred_color, zorder=3)  # Reduced linewidth

        err = w_pred[:, i] - w_target[:, i]
        mae_i = np.mean(np.abs(err))
        rmse_i = np.sqrt(np.mean(err ** 2))

        axes[i].set_title(f"{omega_labels[i]} |  MAE={mae_i:.4e}, RMSE={rmse_i:.4e}")
        axes[i].set_ylabel(f"{omega_labels[i]}", fontsize=14)
        axes[i].legend(loc="upper right", frameon=True, fontsize=14, facecolor="white")
        axes[i].tick_params(axis='both', labelsize=12)
    axes[-1].set_xlabel("Time (s)", fontsize=14)
    fig.suptitle("Model prediction example: $\\omega$", fontsize=14, color=C_NAVY)
    save_path = os.path.join(FIG_DIR, "fig3b_prediction_omega.svg")
    fig.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {save_path}")


def print_summary(summary_df, phase_df=None):
    print("\n================ Summary statistics ================\n")
    print(summary_df.to_string(index=False))

    if phase_df is not None:
        print("\n================ Training phase summary ================\n")
        print(phase_df.to_string(index=False))


# -----------------------
# Main
# -----------------------
def main():
    training_history_csv = os.path.join(OUT_DIR, "training_history.csv")
    phase_summary_csv = os.path.join(OUT_DIR, "training_phase_summary.csv")
    prediction_timeseries_csv = os.path.join(OUT_DIR, "prediction_timeseries.csv")
    prediction_errors_csv = os.path.join(OUT_DIR, "prediction_errors.csv")
    prediction_summary_csv = os.path.join(OUT_DIR, "prediction_summary_statistics.csv")

    require_file(training_history_csv)
    require_file(prediction_timeseries_csv)
    require_file(prediction_errors_csv)
    require_file(prediction_summary_csv)

    history_df = pd.read_csv(training_history_csv)
    pred_df = pd.read_csv(prediction_timeseries_csv)
    errors_df = pd.read_csv(prediction_errors_csv)
    summary_df = pd.read_csv(prediction_summary_csv)
    phase_df = pd.read_csv(phase_summary_csv) if os.path.exists(phase_summary_csv) else None

    plot_training_convergence(history_df)
    plot_error_metrics(errors_df)
    plot_prediction_examples(pred_df)
    print_summary(summary_df, phase_df)

    print(f"\nReplotted figures are saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()
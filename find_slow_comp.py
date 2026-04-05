import pandas as pd
import matplotlib.pyplot as plt

from frequency_est import compute_fft
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter
import os

# For quaternion averaging
from quaternion_ave import average_quaternions_eigen

# Path to your uploaded file
date = "0204"
folder_name = f"./gprbag/gpr_{date}"



file_path = f"{folder_name}/mavros-imu-data.csv"

# Read CSV
df = pd.read_csv(file_path)

# Extract columns into lists
time = df["Time"].tolist()

time = [t - time[0] for t in time]
q_w = df["orientation.w"].tolist()
q_x = df["orientation.x"].tolist()
q_y = df["orientation.y"].tolist()
q_z = df["orientation.z"].tolist()

ang_vel_x = df["angular_velocity.x"].tolist()
ang_vel_y = df["angular_velocity.y"].tolist()
ang_vel_z = df["angular_velocity.z"].tolist()

lin_acc_x = df["linear_acceleration.x"].tolist()
lin_acc_y = df["linear_acceleration.y"].tolist()
lin_acc_z = df["linear_acceleration.z"].tolist()

# Compute sampling frequency
time_diffs = [time[i+1] - time[i] for i in range(len(time)-1)]
avg_time_diff = sum(time_diffs) / len(time_diffs)
sampling_freq = 1 / avg_time_diff

print(f"Sampling frequency: {sampling_freq} Hz")

fft_sliding_window = np.ceil(2.0 * sampling_freq)

# Compute frequency for each time instant using centered sliding window
window_size = int(fft_sliding_window)
freq_estimates = []


# Use compute_fft correctly: pass timestamps and signal window, extract dominant frequency
for i in range(len(lin_acc_z)):
    # Define centered window bounds
    start_idx = max(0, i - window_size // 2)
    end_idx = min(len(lin_acc_z), i + window_size // 2 + 1)

    # Extract window for both time and signal
    window_signal = lin_acc_z[start_idx:end_idx]
    window_time = time[start_idx:end_idx]

    # print(f"window_signal size: {len(window_signal)}, window_time size: {len(window_time)}")

    # Only compute if window is large enough
    if len(window_signal) > 3:
        freqs, magnitude = compute_fft(window_time, window_signal)
        if len(magnitude) > 0:
            valid_idx = np.where((freqs >= 1.0) & (freqs <= 8.0))[0]
            if len(valid_idx) > 0:
                valid_freqs = freqs[valid_idx]
                valid_magnitude = magnitude[valid_idx]
                weight_from_mag = valid_magnitude / np.sum(valid_magnitude)
                # Compute weighted mean with magnitude as weights
                dominant_freq = np.sum(valid_freqs * weight_from_mag)
            else:
                dominant_freq = 0.0
        else:
            dominant_freq = 0.0
    else:
        dominant_freq = 0.0
    freq_estimates.append(dominant_freq)

# Initialize Kalman filter
kf = KalmanFilter(dim_x=1, dim_z=1)
kf.x = np.array([[0.]])  # initial state
kf.F = np.array([[1.]])  # state transition
kf.H = np.array([[1.]])  # measurement function
kf.P = np.array([[1.]])  # covariance matrix
kf.R = np.array([[2.0]])  # measurement noise (increased for more smoothing)
kf.Q = np.array([[0.001]])  # process noise (decreased for smoother predictions)

# Apply Kalman filter
freq_estimates_filtered = []
for freq in freq_estimates:
    kf.predict()
    kf.update(freq)
    freq_estimates_filtered.append(kf.x[0, 0])

freq_estimates = freq_estimates_filtered

# Plot frequency estimates over time
fig, axs = plt.subplots(4, 1, figsize=(12, 12))

# Plot 1: Quaternion components
axs[0].plot(time, q_w, label="q_w")
axs[0].plot(time, q_x, label="q_x")
axs[0].plot(time, q_y, label="q_y")
axs[0].plot(time, q_z, label="q_z")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Quaternion")
axs[0].set_title("Quaternion Components")
axs[0].legend()
axs[0].grid()

# Plot 2: Angular velocity
axs[1].plot(time, ang_vel_x, label="ang_vel_x")
axs[1].plot(time, ang_vel_y, label="ang_vel_y")
axs[1].plot(time, ang_vel_z, label="ang_vel_z")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Angular Velocity (rad/s)")
axs[1].set_title("Angular Velocity")
axs[1].legend()
axs[1].grid()

# Plot 3: Linear acceleration
axs[2].plot(time, lin_acc_x, label="lin_acc_x")
axs[2].plot(time, lin_acc_y, label="lin_acc_y")
axs[2].plot(time, lin_acc_z, label="lin_acc_z")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Linear Acceleration (m/s²)")
axs[2].set_title("Linear Acceleration")
axs[2].legend()
axs[2].grid()

# Plot 4: Estimated frequency
axs[3].plot(time, freq_estimates, label="Estimated Frequency (Hz)")
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel("Frequency (Hz)")
axs[3].set_title("Estimated Frequency Over Time")
axs[3].legend()
axs[3].grid()

# Link x-axes for synchronized zooming
axs[1].sharex(axs[0])
axs[2].sharex(axs[0])
axs[3].sharex(axs[0])

plt.tight_layout()
plt.show()



# --- Compute q_slow (cycle-averaged quaternion) ---
# 1. Use estimated frequency to get period T at each time
# 2. For each time index, extract quaternion samples in window [t-T/2, t+T/2]
# 3. Enforce sign consistency, then average using principal eigenvector

q_slow_list = []
q_array = np.stack([q_w, q_x, q_y, q_z], axis=1)  # shape (N, 4)
for i, freq in enumerate(freq_estimates):
    if i % 100 == 0:
        print(f"Processing time index {i}/{len(time)}")
    # Get period T at this time (avoid zero freq)
    freq = freq if freq > 1e-3 else 1.0
    T = 1.0 / freq
    # Convert period to number of samples
    window_samples = int(np.ceil(T * sampling_freq))
    # Find indices in window [i - window_samples//2, i + window_samples//2]
    start_idx = max(0, i - window_samples // 2)
    end_idx = min(len(q_array), i + window_samples // 2 + 1)
    
    if end_idx - start_idx < 2:
        # Not enough samples, fallback to current quaternion
        q_slow_list.append(q_array[i])
        continue
    
    quats_window = q_array[start_idx:end_idx]
    
    # Enforce sign consistency: align all quaternions to the first one
    ref_quat = quats_window[0]
    for j in range(len(quats_window)):
        if np.dot(quats_window[j], ref_quat) < 0:
            quats_window[j] = -quats_window[j]
    
    # Average using eigenvector method
    q_slow = average_quaternions_eigen(quats_window)
    # Enforce sign consistency for q_slow with respect to the original quaternion at index i
    ref_q = q_array[i]
    if np.dot(q_slow, ref_q) < 0:
        q_slow = -q_slow
    q_slow_list.append(q_slow)

q_slow_array = np.stack(q_slow_list, axis=0)  # shape (N, 4)

# --- Compute omega_slow (cycle-averaged angular velocity) ---
omega_array = np.stack([ang_vel_x, ang_vel_y, ang_vel_z], axis=1)  # shape (N, 3)
omega_slow_list = []

for i, freq in enumerate(freq_estimates):
    # Get period T at this time (avoid zero freq)
    freq = freq if freq > 1e-3 else 1.0
    T = 1.0 / freq
    # Convert period to number of samples
    window_samples = int(np.ceil(T * sampling_freq))
    # Find indices in window [i - window_samples//2, i + window_samples//2]
    start_idx = max(0, i - window_samples // 2)
    end_idx = min(len(omega_array), i + window_samples // 2 + 1)
    
    if end_idx - start_idx < 2:
        # Not enough samples, fallback to current angular velocity
        omega_slow_list.append(omega_array[i])
        continue
    
    omega_window = omega_array[start_idx:end_idx]
    # Average angular velocity over the window
    omega_slow = np.mean(omega_window, axis=0)
    omega_slow_list.append(omega_slow)

omega_slow_array = np.stack(omega_slow_list, axis=0)  # shape (N, 3)


# Optionally, plot q_slow components
fig2, axs2 = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
labels = ["q_slow_w", "q_slow_x", "q_slow_y", "q_slow_z"]
for j in range(4):
    axs2[j].plot(time, q_slow_array[:, j], label=labels[j], color="tab:blue", zorder=3)
    axs2[j].plot(time, [q_w, q_x, q_y, q_z][j], label=f"q_{['w', 'x', 'y', 'z'][j]}", color="tab:red")
    # axs2[j].plot(time, q_slow_array[:, j], label=labels[j])
    axs2[j].set_ylabel(labels[j])
    axs2[j].legend()
    axs2[j].grid()
axs2[-1].set_xlabel("Time (s)")
axs2[0].set_title("Cycle-averaged Quaternion Components (q_slow)")
plt.tight_layout()
plt.show()

fig3, axs3 = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
labels = ["omega_slow_x", "omega_slow_y", "omega_slow_z"]
for j in range(3):
    axs3[j].plot(time, omega_slow_array[:, j], label=labels[j], color="tab:blue", zorder=3)
    axs3[j].plot(time, [ang_vel_x, ang_vel_y, ang_vel_z][j], label=f"omega_{['x', 'y', 'z'][j]}", color="tab:red")
    axs3[j].set_ylabel(labels[j])
    axs3[j].legend()
    axs3[j].grid()
axs3[-1].set_xlabel("Time (s)")
axs3[0].set_title("Cycle-averaged Angular Velocity (omega_slow)")
plt.tight_layout()
plt.show()

# --- Extract data between start and end time instants ---
start_time = float(input("Enter start time (s): "))
end_time = float(input("Enter end time (s): "))

# Find indices corresponding to the start and end time
start_idx = next(i for i, t in enumerate(time) if t >= start_time)
end_idx = next(i for i, t in enumerate(time) if t > end_time)

# Extract data within the specified time range
q_segment = q_array[start_idx:end_idx]
omega_segment = np.stack([ang_vel_x, ang_vel_y, ang_vel_z], axis=1)[start_idx:end_idx]
q_slow_segment = q_slow_array[start_idx:end_idx]
omega_slow_segment = omega_slow_array[start_idx:end_idx]
time_segment = np.array(time[start_idx:end_idx])




# Store the extracted segments
# folder_name = "./output_segments"
output_df = pd.DataFrame({
    'time': time_segment,
    'q_w': q_segment[:, 0],
    'q_x': q_segment[:, 1],
    'q_y': q_segment[:, 2],
    'q_z': q_segment[:, 3],
    'omega_x': omega_segment[:, 0],
    'omega_y': omega_segment[:, 1],
    'omega_z': omega_segment[:, 2],
    'q_slow_w': q_slow_segment[:, 0],
    'q_slow_x': q_slow_segment[:, 1],
    'q_slow_y': q_slow_segment[:, 2],
    'q_slow_z': q_slow_segment[:, 3],
    'omega_slow_x': omega_slow_segment[:, 0],
    'omega_slow_y': omega_slow_segment[:, 1],
    'omega_slow_z': omega_slow_segment[:, 2]
})

output_df.to_csv(f"segment_data_{date}.csv", index=False)
print(f"Data saved to segment_data_{date}.csv")
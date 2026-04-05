import numpy as np
import pandas as pd

from logmap_demon import quat_to_rotvec_log

# Load segment data
df = pd.read_csv("segment_data_0204.csv")

# Extract quaternions: format is [w, x, y, z]
q_array = df[["q_w", "q_x", "q_y", "q_z"]].to_numpy()
q_slow_array = df[["q_slow_w", "q_slow_x", "q_slow_y", "q_slow_z"]].to_numpy()
# Extract time and angular velocity
time = df["time"].to_numpy()
angular_vel = df[["omega_x", "omega_y", "omega_z"]].to_numpy()

# Convert each quaternion to rotation vector via logarithmic map
rotvec = np.array([quat_to_rotvec_log(q) for q in q_array])
rotvec_slow = np.array([quat_to_rotvec_log(q) for q in q_slow_array])

print(f"Loaded {len(df)} samples from segment_data_1228_2.csv")
print(f"rotvec shape:      {rotvec.shape}")
print(f"rotvec_slow shape: {rotvec_slow.shape}")
print(f"\nFirst 3 rotvec rows:\n{rotvec[:3]}")
print(f"\nFirst 3 rotvec_slow rows:\n{rotvec_slow[:3]}")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot rotvec over time
axes[0].plot(rotvec)
axes[0].set_ylabel("Rotation Vector")
axes[0].set_title("Rotation Vector vs Time")
axes[0].legend(["x", "y", "z"])
axes[0].grid(True)

# Plot rotvec_slow over time
axes[1].plot(rotvec_slow)
axes[1].set_ylabel("Slow Rotation Vector")
axes[1].set_title("Slow Rotation Vector vs Time")
axes[1].legend(["x", "y", "z"])
axes[1].grid(True)

# Plot angular velocity over time
axes[2].plot(angular_vel)
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Angular Velocity")
axes[2].set_title("Angular Velocity vs Time")
axes[2].legend(["x", "y", "z"])
axes[2].grid(True)


axes[1].sharex(axes[0])
axes[2].sharex(axes[0])



# Link the x-axes of all subplots for synchronized zooming
plt.tight_layout()
plt.show()
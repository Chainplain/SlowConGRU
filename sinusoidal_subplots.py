import numpy as np
import matplotlib.pyplot as plt


# Time axis
x = np.linspace(0, 10, 2000)

# Components
slow_component = np.sin(2 * np.pi * 0.15 * x) + 0.5 * np.sin(2 * np.pi * 0.3 * x) + 0.25 * np.cos(2 * np.pi * 0.5 * x)
fast_component = 0.3 * np.sin(2 * np.pi * 4.5 * x)

# Combined curve
combined_curve = slow_component + fast_component

# Create horizontal subplots
fig, axes = plt.subplots(1, 2, figsize=(6, 4))

# Left: combined curve
axes[0].plot(x, combined_curve, linewidth=2)
axes[0].set_title("Combined Curve")
axes[0].grid(False)
axes[0].set_xticks([])
axes[0].set_yticks([])

# Right: slow component only
axes[1].plot(x, slow_component, linewidth=2, color="#40E6A4")
axes[1].set_title("Slow Component")
axes[1].grid(False)
axes[1].set_xticks([])
axes[1].set_yticks([])

fig.savefig('./combined_curve.svg', format='svg', bbox_inches='tight')
axes[1].figure.savefig('./slow_component.svg', format='svg', bbox_inches='tight')
plt.tight_layout()
plt.show()

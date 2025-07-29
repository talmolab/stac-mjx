import mujoco
import numpy as np
import stac_mjx
from pathlib import Path
import h5py

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import time
import imageio
import jax.numpy as jnp

# Enable XLA flags if on GPU
stac_mjx.enable_xla_flags()

# Choose parent directory as base path for data files
base_path = Path.cwd()
# Load configs
cfg = stac_mjx.load_configs(base_path / "configs")

# Load data
kp_data_scaled, sorted_kp_names = stac_mjx.load_mocap(cfg, base_path)

# Assume kp_data_scaled is the result of your previous JAX operations,
# with shape [num_frames, total_flattened_dim] where total_flattened_dim = xyz * len(model_inds)

# --- Important: You need to know the original number of keypoints (len(model_inds)) ---
# Let's assume you have a variable for this, e.g., `num_selected_keypoints`
# For demonstration, let's derive it from the total flattened dimension, assuming xyz=3
# In your actual code, `len(model_inds)` would be a known value.
num_selected_keypoints = kp_data_scaled.shape[1] // 3 # Divide by 3 (x,y,z)

print(num_selected_keypoints)

print(f"Shape: {kp_data_scaled.shape}")
print(f"Min: {kp_data_scaled.min()}, Max: {kp_data_scaled.max()}, Mean: {kp_data_scaled.mean()}")
print(f"First frame sample (flattened):\n{kp_data_scaled[0]}")

# --- Setup 3D plot ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Get the initial frame data
initial_frame_flattened = kp_data_scaled[0]

# Reshape the initial frame to separate X, Y, Z for each keypoint
# The flattened data is [X1..XN, Y1..YN, Z1..ZN]
# So we first reshape to [xyz, num_selected_keypoints]
# Then transpose to get [num_selected_keypoints, xyz]
initial_frame_reshaped_for_plot = initial_frame_flattened.reshape(3, num_selected_keypoints).T
x_init, y_init, z_init = initial_frame_reshaped_for_plot[:, 0], initial_frame_reshaped_for_plot[:, 1], initial_frame_reshaped_for_plot[:, 2]
print(f"X: {x_init}\n")
print(f"Y: {y_init}\n")
print(f"Z: {z_init}\n")
scatter = ax.scatter(x_init, y_init, z_init, c="b", marker="o")

# Axis labels
ax.set_xlabel("X (scaled)")
ax.set_ylabel("Y (scaled)")
ax.set_zlabel("Z (scaled)")

# Axis limits with buffer
# When calculating min/max, we need to consider the reshaped structure for all frames
# We know that for each flattened frame:
# First `num_selected_keypoints` values are X coordinates
# Next `num_selected_keypoints` values are Y coordinates
# Next `num_selected_keypoints` values are Z coordinates
min_x = kp_data_scaled[:, :num_selected_keypoints].min()
max_x = kp_data_scaled[:, :num_selected_keypoints].max()
min_y = kp_data_scaled[:, num_selected_keypoints : 2 * num_selected_keypoints].min()
max_y = kp_data_scaled[:, num_selected_keypoints : 2 * num_selected_keypoints].max()
min_z = kp_data_scaled[:, 2 * num_selected_keypoints : 3 * num_selected_keypoints].min()
max_z = kp_data_scaled[:, 2 * num_selected_keypoints : 3 * num_selected_keypoints].max()

# It's often good practice to make the plot limits somewhat symmetrical or centered
# You can adjust these based on your data's typical range
buffer = 5 # Add a fixed buffer for visualization
# ax.set_xlim(min_x - buffer, max_x + buffer)
# ax.set_ylim(min_y - buffer, max_y + buffer)
# ax.set_zlim(min_z - buffer, max_z + buffer)
ax.set_xlim(-30, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(-30, 30)
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

# --- Manual animation loop ---
title_text = ax.set_title("3D Keypoints - Frame 0")
num_frames = kp_data_scaled.shape[0]

# Iterate through a reasonable number of frames, not more than available
for frame_idx in range(min(70, num_frames)): # Added min to prevent index out of bounds if num_frames < 70
    frame_data_flattened = kp_data_scaled[frame_idx]

    # Reshape the current frame for plotting, same as initial_frame
    frame_data_reshaped_for_plot = frame_data_flattened.reshape(3, num_selected_keypoints).T
    x, y, z = frame_data_reshaped_for_plot[:, 0], frame_data_reshaped_for_plot[:, 1], frame_data_reshaped_for_plot[:, 2]

    scatter._offsets3d = (x, y, z)
    title_text.set_text(f"3D Keypoints - Frame {frame_idx}/{num_frames - 1}")

    # Capture the frame as an image (ARGB) - if you intend to save an animation,
    # you'd collect these frames and then use something like imageio or matplotlib.animation
    fig.canvas.draw()

    plt.pause(0.05)  # Adjust pause for animation speed (e.g., 0.033 for ~30 FPS, 0.05 for ~20 FPS)
    print(f"Frame {frame_idx} displayed")

plt.show()

# Close the plot to prevent it from hanging if you only wanted to save
plt.close(fig)
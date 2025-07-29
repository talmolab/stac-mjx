import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import time
import imageio

# --- Load keypoints CSV ---
file_path = "/Users/rusham/Documents/TalmoLab/GitRepos/stac-keypoints-ui/stac_keypoints_ui/data/014_h_offset.csv"
df = pd.read_csv(file_path)

# --- Extract coordinate columns (ending in _x/_y/_z) ---
kp_coord_cols = df.filter(regex='_[xyz]$', axis=1).columns.tolist()

# --- Parse keypoint base names ---
kp_base_names = [re.sub(r'_[xyz]$', '', col) for col in kp_coord_cols]
kp_names_ordered = []
seen = set()
for name in kp_base_names:
    if name not in seen:
        kp_names_ordered.append(name)
        seen.add(name)

# --- Build ordered list of coord columns: [kp1_x, kp1_y, kp1_z, kp2_x, ...] ---
sorted_kp_coord_cols = []
for name in kp_names_ordered:
    sorted_kp_coord_cols.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

# --- Extract and reshape data ---
actual_kp_df = df[sorted_kp_coord_cols]
flat_kp_data = actual_kp_df.values.astype(np.float32)
num_frames = flat_kp_data.shape[0]
num_keypoints = len(kp_names_ordered)

assert actual_kp_df.shape[1] == num_keypoints * 3, \
    f"Expected {num_keypoints * 3} cols, got {actual_kp_df.shape[1]}"

kp_data_reshaped = flat_kp_data.reshape(num_frames, num_keypoints, 3)

# --- Scale data ---
VISUALIZATION_SCALE_FACTOR = 1
kp_data_scaled = kp_data_reshaped * VISUALIZATION_SCALE_FACTOR

print(f"Shape: {kp_data_scaled.shape}")
print(f"Min: {kp_data_scaled.min()}, Max: {kp_data_scaled.max()}, Mean: {kp_data_scaled.mean()}")
#print(f"First frame sample:\n{kp_data_scaled[0]}")

# --- Setup 3D plot ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

initial_frame = kp_data_scaled[0]
x_init, y_init, z_init = initial_frame[:, 0], initial_frame[:, 1], initial_frame[:, 2]
print(f"X: {x_init}\n")
print(f"Y: {y_init}\n")
print(f"Z: {z_init}\n")

scatter = ax.scatter(x_init, y_init, z_init, c="b", marker="o")

# Axis labels
ax.set_xlabel("X (scaled)")
ax.set_ylabel("Y (scaled)")
ax.set_zlabel("Z (scaled)")

# Axis limits with buffer
min_x, max_x = kp_data_scaled[:, :, 0].min(), kp_data_scaled[:, :, 0].max()
min_y, max_y = kp_data_scaled[:, :, 1].min(), kp_data_scaled[:, :, 1].max()
min_z, max_z = kp_data_scaled[:, :, 2].min(), kp_data_scaled[:, :, 2].max()
range_max = max(max_x - min_x, max_y - min_y, max_z - min_z)
buffer = 0.1 * range_max if range_max > 0 else 1.0

ax.set_xlim(-30, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(-30, 30)
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

# --- Manual animation loop ---
title_text = ax.set_title("3D Keypoints - Frame 0")
output_video_path = "keypoints_animation.mp4"
fps = 10 # Frames per second for the output video (adjust as desired)
frames_for_video = []

for frame_idx in range(70):
    frame_data = kp_data_scaled[frame_idx]
    x, y, z = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
    scatter._offsets3d = (x, y, z)
    title_text.set_text(f"3D Keypoints - Frame {frame_idx}/{num_frames - 1}")

    # Capture the frame as an image (ARGB)
    fig.canvas.draw()
    # Change to tostring_argb()
    image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # Convert ARGB to RGBA if imageio expects RGBA (which it typically does)
    # The order of channels for `tostring_argb()` is Alpha, Red, Green, Blue.
    # We need Red, Green, Blue, Alpha.
    # So, swap the 0th channel (Alpha) to the last position.
    image = np.roll(image, shift=-1, axis=-1) # Shifts the last axis by -1 (A R G B -> R G B A)

    frames_for_video.append(image)

    plt.pause(0.1)  # ~30 FPS
    print(f"Frame {frame_idx} displayed")

plt.show()

# --- Save video ---
print(f"Saving video to {output_video_path}...")
imageio.mimsave(output_video_path, frames_for_video, fps=fps)
print("Video saved successfully!")

# Close the plot to prevent it from hanging if you only wanted to save
plt.close(fig)
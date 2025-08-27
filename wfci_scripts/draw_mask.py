import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage.draw import polygon2mask
from PIL import Image
import os

# === Load image ===
img_path = "mean_image_for_mask.png"  # Make sure this exists in your working dir
img = np.array(Image.open(img_path).convert('L'))  # Convert to grayscale if needed
height, width = img.shape

# === Plot and collect polygon points ===
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
ax.set_title("Click to define brain mask polygon.\nClose window when done.")

# Collect polygon vertices with ginput
pts = plt.ginput(n=-1, timeout=0)  # infinite points until closed manually
plt.close()

# Convert to NumPy array (as (row, col) format)
pts_array = np.array([(y, x) for x, y in pts])  # (y, x) because image coords are (row, col)

# === Create mask ===
mask = polygon2mask((height, width), pts_array).astype(np.uint8)

# === Save the mask ===
out_path = "brain_mask.npy"
print("Saving mask...")  # Add this
np.save(out_path, mask)
print(f"✅ Mask saved to {out_path}")  # Final confirmation

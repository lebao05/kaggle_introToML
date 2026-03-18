import numpy as np

# Load file
data = np.load("./23127321_23127325/data/pixels.npz")

# Check keys
print("Keys:", data.files)

# Get pixels array
pixels = data["pixels"]

# Print shape and dtype
print("Shape:", pixels.shape)
print("Dtype:", pixels.dtype)

# First 10 rows
print("First 10 rows:")
print(pixels[:1000])
from pathlib import Path
from fur_dataset import FurDataset, BufferType
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
import torch

torch.manual_seed(48)  # For reproducibility of random transformations

# Add RandomCrop to your transforms
# We use a size like 256 or 128 to verify the cropping logic
transforms = T.Compose([
    T.RandomRotation(degrees=(-180, 180)),
    T.RandomCrop(size=(256, 256)),
])

base_path = Path("data/synthetic_fur_images/images/with_ground_truth")

dataset = FurDataset(
    base_path=base_path,
    scenes=["bunny_3Lights_rotateUp", "bunny_3Lights_static"], 
    buffer_types=[BufferType.Rasterized, BufferType.SceneDepth], 
    transforms=transforms
)

# Get a sample
sample = dataset[100] 
inputs = sample['bufferStack']
target = sample['target']

print(f"Input Shape: {inputs.shape}")   # Expect [4, 256, 256] (3 RGB + 1 Depth)
print(f"Target Shape: {target.shape}")  # Expect [3, 256, 256]

# Extract specific buffers for visualization
# Assuming order: Rasterized (3 ch) -> SceneDepth (1 ch)
rasterized = inputs[:3, :, :]
scene_depth = inputs[3:4, :, :] # Keep dimension for now

# Visualize Side-by-Side to check alignment
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot Rasterized Input
ax[0].imshow(rasterized.permute(1, 2, 0).numpy())
ax[0].set_title("Input: Rasterized (Cropped)")
ax[0].axis('off')

# Plot Depth Input
# Squeeze the 1st dimension to make it (H, W) for grayscale plotting
ax[1].imshow(scene_depth.squeeze(0).numpy(), cmap='gray') 
ax[1].set_title("Input: Depth (Cropped)")
ax[1].axis('off')

# Plot Ground Truth Target
ax[2].imshow(target.permute(1, 2, 0).numpy())
ax[2].set_title("Target: High Quality (Cropped)")
ax[2].axis('off')

plt.tight_layout()
plt.show()
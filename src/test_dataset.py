from pathlib import Path
from fur_dataset import FurDataset, BufferType

import matplotlib.pyplot as plt

import torchvision.transforms.v2 as T

transforms = T.Compose([
    T.RandomRotation(degrees=(-180, 180)),
    # T.ColorJitter(brightness=20, contrast=0.2, saturation=0.2, hue=0.1),
    ])

base_path=Path("data/synthetic_fur_images/images/with_ground_truth")

test = FurDataset(
        base_path=base_path,
        scenes=["bunny_3Lights_rotateUp", "bunny_3Lights_static"], 
        buffer_types=[BufferType.Rasterized, BufferType.SceneDepth], 
        transforms=transforms
    )

for key, value in test[0].items():
    print(f"  {key}: {value.shape}")
    # show image of depth buffer
    if key == "bufferStack":
        # Split bufferstack into rasterized and scene depth
        rasterized = value[:3, :, :]
        scene_depth = value[3, :, :]
        # Show rasterized buffer
        plt.imshow(rasterized.permute(1, 2, 0).numpy())
        plt.title("Rasterized Buffer")
        plt.axis('off') 

        plt.show()

        # Show scene depth buffer   
        plt.imshow(scene_depth.numpy(), cmap='gray')
        plt.title("Scene Depth Buffer")
        plt.axis('off')
        plt.show()
        
    else:
        plt.imshow(value.permute(1, 2, 0).numpy())
        plt.title(key)
        plt.axis('off')        
        plt.show()
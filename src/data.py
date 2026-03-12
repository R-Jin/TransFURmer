from pathlib import Path
from torch.utils.data import DataLoader, random_split
from fur_dataset import FurDataset, BufferType
import torchvision.transforms.v2 as T

# All sphere scenes (12)
sphere_scenes = [
    "sphere_3Lights_moveCW",
    "sphere_3Lights_moveRL",
    "sphere_3Lights_sbPlateRL_static",
    "sphere_3Lights_sbPlateRotateUp_static",
    "sphere_3Lights_static",
    "sphere_fillLight_static",
    "sphere_hdriCapeHill_static",
    "sphere_hdriHansaplatz_static",
    "sphere_hdriMalibu_static",
    "sphere_hdriTopanga_static",
    "sphere_keyLight_static",
    "sphere_rimLight_static"
]

# All torus scenes (8)
torus_scenes = [
    "torus_3Lights_moveCW",
    "torus_3Lights_moveRL",
    "torus_3Lights_rotateRightForward",
    "torus_3Lights_rotateUp",
    "torus_3Lights_static60",
    "torus_curly_clumpLarge_brown_3Lights_rotateRightForward",
    "torus_curly_clumpLarge_brown_3Lights_rotateUp",
    "torus_curly_clumpLarge_brown_3Lights_static60"
]

# Tube scenes (2)
tube_scenes = [
    "tube_3Lights_rotateRight",
    "tube_3Lights_static"
]

# Bunny scenes (3)
bunny_scenes = [
    "bunny_3Lights_rotateUp",      # 360 frames
    "bunny_3Lights_static",        # 720 frames
    "bunny_curly_clumpLarge_brown_3Lights_static"  # 720 frames
]

# Split
train_scenes = sphere_scenes + torus_scenes + [tube_scenes[0]]
val_scenes = [tube_scenes[1]]
test_scenes = bunny_scenes

# Creating dataset and dataloader
base_path = Path("data/synthetic_fur_images/images/with_ground_truth")

crop_size = (128, 128)

train_transforms = T.Compose([
    T.RandomCrop(size=crop_size),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
])

val_transforms = T.Compose([
    T.CenterCrop(size=crop_size)
])

test_transforms = T.Compose([
    T.CenterCrop(size=crop_size)
])

buffer_types = [BufferType.SceneDepth, BufferType.LitPrimitive, BufferType.GuideColored, BufferType.WorldNormal, BufferType.Mask]

train_dataset = FurDataset(
    base_path=base_path,
    scenes=train_scenes, 
    buffer_types=buffer_types, 
    transforms=train_transforms
)

val_dataset = FurDataset(
    base_path=base_path,
    scenes=val_scenes, 
    buffer_types=buffer_types, 
    transforms=val_transforms
)

test_dataset = FurDataset(
    base_path=base_path,
    scenes=test_scenes, 
    buffer_types=buffer_types, 
)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
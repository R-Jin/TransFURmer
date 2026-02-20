from models.network_swinir import SwinIR
from fur_dataset import FurDataset, BufferType
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import torch
from torch import nn

base_path = Path("data/synthetic_fur_images/images/with_ground_truth")

crop_size = (128, 128)

transforms = T.Compose([
    T.RandomCrop(size=crop_size)

    # Add more transforms as needed
])

fur_dataset = FurDataset(
    base_path=base_path,
    scenes=["bunny_3Lights_rotateUp"], 
    buffer_types=[BufferType.Rasterized, BufferType.SceneDepth], 
    transforms=transforms
)

dataloader = DataLoader(fur_dataset, batch_size=2, shuffle=True)  # num_workers

batch = next(iter(dataloader))
inputs = batch['bufferStack']
target = batch['target']

print(f"Batch Input Shape: {inputs.shape}")   # Expect [B, C_total, H, W]
print(f"Batch Target Shape: {target.shape}")  # Expect [B, 3

######## RGB Head ###############
class SwinIR_4to3(nn.Module):
    def __init__(self, swinir):
        super().__init__()
        self.swinir = swinir
        self.rgb_head = nn.Conv2d(
            in_channels=4,
            out_channels=3,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        x = self.swinir(x)   # [B,4,H,W]
        x = self.rgb_head(x) # [B,3,H,W]
        return x
####################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

swin = SwinIR(
    upscale=1,
    in_chans=inputs.shape[1],
    img_size=crop_size[0],
    window_size=8,
    img_range=1.0,
    depths=[6, 6],
    embed_dim=60,
    num_heads=[6, 6],
    mlp_ratio=2,
).to(device)

model = SwinIR_4to3(swin)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

model = model.to(device)
inputs = inputs.to(device)

print(f"Model created on {device}")

print("Running a forward pass...")
with torch.no_grad():
    outputs = model(inputs)

print(f"Output Shape: {outputs.shape}")  # Expect [B, 3, H, W]

if outputs.shape == target.shape:
    print("Output shape matches target shape.")
else:
    print("SHAPE MISMATCH: Output shape does not match target shape.")


##### DEBUG SHOW IMAGE #####
import os
import torch
import torchvision.utils as vutils

os.makedirs("debug_outputs", exist_ok=True)

def save_images(pred, target, step):
    # Take first image in batch
    pred_img = pred[0].detach().cpu().clamp(0,1)
    gt_img   = target[0].detach().cpu().clamp(0,1)

    # Stack them vertically: [2,3,H,W]
    stack = torch.stack([pred_img, gt_img], dim=0)

    # Save
    vutils.save_image(
        stack,
        f"debug_outputs/step_{step:04d}.png",
        nrow=1
    )
    
############################

for step in range(1001):
    optimizer.zero_grad()

    pred = model(inputs)
    loss = criterion(pred, target)

    loss.backward()
    optimizer.step()

    # if step % 100 == 0:
    print(step, loss.item())
    if step % 100 == 0:
        save_images(pred, target, step)
        # plot that shit
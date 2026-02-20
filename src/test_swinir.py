from models.network_swinir import SwinIR
from fur_dataset import FurDataset, BufferType
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import torch

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

'''
img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
'''

model = SwinIR(
    upscale=1,
    in_chans=inputs.shape[1],
    img_size=crop_size[0],
    window_size=8,
    img_range=1.0,
    depths=[6, 6],
    embed_dim=60,
    num_heads=[6, 6],
    mlp_ratio=2,
)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
inputs = inputs.to(device)

model.train()

for step in range(2000):
    optimizer.zero_grad()
    
    pred = model(inputs)
    loss = criterion(pred, target)
    
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(step, loss.item())
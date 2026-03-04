from vgg import VGGPerceptualLoss
import csv
from fur_dataset import FurDataset, BufferType
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.v2 as T
import torch
from torch.amp import autocast, GradScaler
from torch import nn
from models.swinir_out3 import SwinIR_out3

EPOCHS = 300

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FREQ = 1

log_file = Path("training_log.csv")
# Create header if it doesn't exist
if not log_file.exists():
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_latest_checkpoint(model, optimizer=None, scheduler=None, scaler=None, checkpoint_dir=CHECKPOINT_DIR, device=device):
    checkpoints = sorted(checkpoint_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not checkpoints:
        print("No checkpoints found. Starting from scratch.")
        return 0  

    latest = checkpoints[0]
    state = torch.load(latest, map_location=device)

    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler and "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    start_epoch = state.get("epoch", -1) + 1
    print(f"Loaded checkpoint '{latest}' (epoch {start_epoch})")
    return start_epoch

# Creating dataset and dataloader
base_path = Path("data/synthetic_fur_images/images/with_ground_truth")

# Sphere = 12 scenes (train) (8640 frames)
# Torus = 8 scenes (train) (5760 frames)
# Tube = 2 scenes (test) (1440 frames)
# Bunny = 3 scenes (validation) (1800 frames)

# sphere + torus = train (14 400 frames)
# tube = test (1440 frames)
# bunny = validation (1800 frames)


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

buffer_types = [BufferType.Rasterized, BufferType.SceneDepth, BufferType.LitPrimitive, BufferType.GuideColored, BufferType.WorldNormal]

train_scenes = [
    "sphere_3Lights_moveCW",
    "sphere_3Lights_moveRL",
    #"sphere_3Lights_sbPlateRL_static",
    #"sphere_3Lights_sbPlateRotateUp_static",
    "sphere_3Lights_static",
    "sphere_fillLight_static",
    #"sphere_hdriCapeHill_static",
    #"sphere_hdriHansaplatz_static",
    #"sphere_hdriMalibu_static",
    #"sphere_hdriTopanga_static",
    "sphere_keyLight_static",
    "sphere_rimLight_static",
    "torus_3Lights_moveCW",
    "torus_3Lights_moveRL",
    "torus_3Lights_rotateRightForward",
    "torus_3Lights_rotateUp",
    "torus_3Lights_static60",
    #"torus_curly_clumpLarge_brown_3Lights_rotateRightForward",
    #"torus_curly_clumpLarge_brown_3Lights_rotateUp",
    #"torus_curly_clumpLarge_brown_3Lights_static60"
]
train_dataset = FurDataset(
    base_path=base_path,
    scenes=train_scenes, 
    buffer_types=buffer_types, 
    transforms=train_transforms
)

val_scenes = [
    "tube_3Lights_rotateRight", "tube_3Lights_static"
]
val_dataset = FurDataset(
    base_path=base_path,
    scenes=val_scenes, 
    buffer_types=buffer_types, 
    transforms=val_transforms
)

test_scenes = [ 
    "bunny_3Lights_rotateUp", 
    "bunny_3Lights_static", 
    "bunny_curly_clumpLarge_brown_3Lights_static"
]

test_dataset = FurDataset(
    base_path=base_path,
    scenes=test_scenes, 
    buffer_types=buffer_types, 
    transforms=test_transforms
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers = 4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers = 4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers = 4, pin_memory=True)

model = SwinIR_out3(
    upscale=1,
    in_chans=train_dataset[0]['bufferStack'].shape[0],
    out_chans=3,
    img_size=crop_size[0],
    window_size=8,
    img_range=1.0,
    depths=[6, 6],
    embed_dim=60,
    num_heads=[6, 6],
    mlp_ratio=2,
).to(device)

# First L1, then VGG perceptual loss
l1_loss_fn = torch.nn.L1Loss() 
vgg_loss_fn = VGGPerceptualLoss(device)
lambda_vgg = 0.1 # weight for perceptual loss
VGG_START_EPOCH = 65

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay = 1e-4, betas=(0.9, 0.99)) 

steps_per_epoch = len(train_dataloader)
total_steps = EPOCHS * steps_per_epoch
warmup_steps = int(0.05 * total_steps)  # 5% warmup

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,   # start at 10% of base LR
    total_iters=warmup_steps 
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max= total_steps - warmup_steps,  # cosine annealing for the remaining 90% of training steps
    eta_min=1e-6
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

scaler = GradScaler(enabled=(device.type == 'cuda'))

start_epoch = load_latest_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    checkpoint_dir=CHECKPOINT_DIR,
    device=device
)

for epoch in range(start_epoch, EPOCHS):
    #### Training ####

    # Set to training mode
    model.train()
    running_loss = 0.0

    # Load data
    for i, sample in enumerate(train_dataloader):
        buffers = sample['bufferStack'].to(device, non_blocking=True)  # [B, num_buffers, H, W]
        target = sample['target'].to(device, non_blocking=True)        # [B, 3, H, W]
        
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            pred = model(buffers)
            l1 = l1_loss_fn(pred, target)

        if epoch >= VGG_START_EPOCH:
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                vgg_loss = vgg_loss_fn(pred, target)

            loss_value = l1 + lambda_vgg * vgg_loss
        else:
            loss_value = l1
    
        # Backward pass
        scaler.scale(loss_value).backward()

        # Gradient clipping (VERY recommended for transformers)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss_value.item()

        if i % 100 == 0 or i == len(train_dataloader)-1:
            print(f"Epoch {epoch} | Batch {i+1}/{len(train_dataloader)} | "
                f"Loss: {running_loss/(i+1):.6f}")
    
    print(f"Epoch {epoch} | Train Loss: {running_loss / len(train_dataloader):.6f}")

    #### Validation ####
    model.eval()
    val_loss = 0.0

    with torch.inference_mode():
        for sample in val_dataloader:
            buffers = sample['bufferStack'].to(device, non_blocking=True)  # [B, num_buffers, H, W]
            target = sample['target'].to(device, non_blocking=True)        # [B, 3, H, W]
            
            # with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            #     pred = model(buffers)
            #     loss_value = loss(pred, target)

            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                pred = model(buffers)
                l1 = l1_loss_fn(pred, target)

            if epoch >= VGG_START_EPOCH:
                with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    vgg_loss = vgg_loss_fn(pred, target)

                loss_value = l1 + lambda_vgg * vgg_loss
            else:
                loss_value = l1
                

            val_loss += loss_value.item()

    val_loss /= len(val_dataloader)
    print(f"Epoch {epoch} | Val Loss: {val_loss:.6f}")

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        current_lr = optimizer.param_groups[0]['lr']
        writer.writerow([epoch, running_loss/len(train_dataloader), val_loss, current_lr])

    if epoch % CHECKPOINT_FREQ == 0:
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }, CHECKPOINT_DIR / f"checkpoint_{epoch}.pth")
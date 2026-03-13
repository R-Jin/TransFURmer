from vgg import VGGPerceptualLoss
import csv
from pathlib import Path
import torchvision.utils as vutils
import torch
from torch.amp import autocast, GradScaler
from models.swinir_out3 import SwinIR_out3
from data import train_dataloader, val_dataloader, test_dataloader, crop_size, train_dataset

EPOCHS = 300

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FREQ = 5

COMPARISON_DIR = Path("comparisons")
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_FREQ = 10  # Save comparison images every N epochs


def save_comparison(pred, target, epoch, n_images=4):
    """Save a side-by-side grid of predicted vs ground truth images."""
    n = min(n_images, pred.shape[0])
    pred_imgs = pred[:n].detach().cpu().clamp(0, 1)
    gt_imgs = target[:n].detach().cpu().clamp(0, 1)
    # Interleave: pred0, gt0, pred1, gt1, ...
    pairs = torch.stack([pred_imgs, gt_imgs], dim=1).flatten(0, 1)  # [2n, C, H, W]
    vutils.save_image(pairs, COMPARISON_DIR / f"epoch_{epoch:04d}.png", nrow=2)

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
        try:
            scheduler.load_state_dict(state["scheduler"])
        except KeyError:
            print("Scheduler state not found in checkpoint. Starting scheduler from scratch.")
    if scaler and "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    start_epoch = state.get("epoch", -1) + 1
    print(f"Loaded checkpoint '{latest}' (epoch {start_epoch})")
    return start_epoch

model = SwinIR_out3(
    upscale=1,
    in_chans=train_dataset[0]['bufferStack'].shape[0],
    out_chans=3,
    img_size=crop_size[0],
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6],
    embed_dim=90,
    num_heads=[6, 6, 6, 6],
    mlp_ratio=4,
).to(device)

# First L1, then VGG perceptual loss
l1_loss_fn = torch.nn.L1Loss() 
vgg_loss_fn = VGGPerceptualLoss(device)
lambda_vgg = 0.3 # weight for perceptual loss
VGG_START_EPOCH = 60

def fft_loss(pred, target, high_freq_weight=2.0):
    # Cast to float32 for FFT (bfloat16 not supported)
    pred = pred.float()
    target = target.float()
    f_pred = torch.fft.rfft2(pred, norm='ortho')
    f_target = torch.fft.rfft2(target, norm='ortho')
    # Create frequency weighting mask
    H, W = pred.shape[-2:]
    u = torch.fft.fftfreq(H, device=pred.device)[:, None].abs()   # [H, 1]
    v = torch.fft.rfftfreq(W, device=pred.device)[None, :].abs()  # [1, W//2+1]
    weight = 1 + high_freq_weight * (u**2 + v**2)  # shape [H, W//2+1]
    # Loss on magnitude spectra
    return (weight * (f_pred.abs() - f_target.abs()).abs()).mean()

lambda_fft = 0.02 # weight for FFT loss

def gradient_loss(pred, target):
    pred = pred.float()
    target = target.float()
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()

lambda_grad = 0.1 # weight for gradient loss

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay = 1e-4, betas=(0.9, 0.99), eps=1e-6) 

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
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            pred = model(buffers)
            l1 = l1_loss_fn(pred, target)
            fft_val = fft_loss(pred, target)
            grad_val = gradient_loss(pred, target)
            loss_value = l1 + lambda_fft * fft_val + lambda_grad * grad_val

        if epoch >= VGG_START_EPOCH:
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                vgg_loss = vgg_loss_fn(pred, target)

            loss_value = loss_value + lambda_vgg * vgg_loss

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
        for i, sample in enumerate(val_dataloader):
            buffers = sample['bufferStack'].to(device, non_blocking=True)  # [B, num_buffers, H, W]
            target = sample['target'].to(device, non_blocking=True)        # [B, 3, H, W]

            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                pred = model(buffers)
                l1 = l1_loss_fn(pred, target)
                fft_val = fft_loss(pred, target)
                grad_val = gradient_loss(pred, target)
                loss_value = l1 + lambda_fft * fft_val + lambda_grad * grad_val

            if epoch >= VGG_START_EPOCH:
                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    vgg_loss = vgg_loss_fn(pred, target)

                loss_value = loss_value + lambda_vgg * vgg_loss

            val_loss += loss_value.item()

            if i == 0 and epoch % COMPARISON_FREQ == 0:
                save_comparison(pred, target, epoch)

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
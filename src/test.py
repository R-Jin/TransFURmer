from pathlib import Path

from models.swinir_out3 import SwinIR_out3

import torch
import torchvision.utils as vutils

from data import test_dataloader, crop_size, train_dataset

CHECKPOINT_DIR = Path("checkpoints")
PATH = "checkpoints/checkpoint_95.pth"
OUTPUT_DIR = Path("stitch_test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
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

print(f"Loading model weights from {PATH}...")
load_latest_checkpoint(model, checkpoint_dir=Path(PATH).parent, device=device)

print("Model loaded successfully.")
model.eval()

def create_slices(x, slice_size=(64, 64)):
    """Return list of (slice_tensor, (i, j)) for non-overlapping tiles."""
    B, C, H, W = x.shape
    slices = []
    for i in range(0, H, slice_size[0]):
        for j in range(0, W, slice_size[1]):
            slice_ = x[:, :, i:i+slice_size[0], j:j+slice_size[1]]
            slices.append((slice_, (i, j)))
    return slices


def stitch_slices(slices_with_pos, out_shape):
    """Reconstruct a full image from slices (B,C,h,w) plus their positions."""
    B, C, H, W = out_shape
    out = torch.zeros(out_shape, device=slices_with_pos[0][0].device)
    for slice_, (i, j) in slices_with_pos:
        _, _, h, w = slice_.shape
        out[:, :, i:i+h, j:j+w] = slice_
    return out


with torch.inference_mode():
    sample = next(iter(test_dataloader))
    buffers = sample["bufferStack"].to(device, non_blocking=True)  # [B, num_buffers, H, W]
    target = sample["target"].to(device, non_blocking=True)        # [B, 3, H, W]

    # Only process the first sample for full-image reconstruction.
    buffers = buffers[0:1]
    target = target[0:1]

    buffer_slices = create_slices(buffers, slice_size=(1024, 1024))
    # (Optional) target_slices = create_slices(target, slice_size=crop_size)

    # Process slices in manageable batches to avoid OOM.
    # Set SLICE_BATCH_SIZE via env var to tune memory usage.
    import os
    slice_batch_size = int(os.getenv("SLICE_BATCH_SIZE", "16"))

    output_full = torch.zeros(target.shape, device=device)
    for start in range(0, len(buffer_slices), slice_batch_size):
        batch_slices = buffer_slices[start:start + slice_batch_size]
        batch_in = torch.cat([s for s, _ in batch_slices], dim=0)
        batch_out = model(batch_in)

        # Write each slice into the full image
        for idx in range(batch_out.shape[0]):
            out_slice = batch_out[idx:idx+1]
            i, j = batch_slices[idx][1]
            h, w = out_slice.shape[2:]
            output_full[:, :, i:i+h, j:j+w] = out_slice

    # Save the prediction and ground truth for inspection
    out_img = output_full.detach().cpu().clamp(0, 1)
    tgt_img = target.detach().cpu().clamp(0, 1)

    vutils.save_image(out_img, OUTPUT_DIR / "output.png")
    vutils.save_image(tgt_img, OUTPUT_DIR / "target.png")
    vutils.save_image(torch.cat([out_img, tgt_img], dim=0), OUTPUT_DIR / "output_vs_target.png", nrow=2)
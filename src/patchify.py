import torch
import time
import torchvision.transforms as transforms

def patchify(img, patch_size):
    # img: [C, H, W]
    C, H, W = img.shape
    ph, pw = patch_size

    assert H % ph == 0 and W % pw == 0

    img = img.reshape(C, H // ph, ph, W // pw, pw)
    img = img.permute(1, 3, 0, 2, 4).contiguous()  # Add contiguous()
    img = img.reshape(-1, C, ph, pw)
    return img

start_time = time.time()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Total time: {elapsed_time:.2f} seconds")
print(f"Time per iteration: {elapsed_time / 60:.2f} seconds")


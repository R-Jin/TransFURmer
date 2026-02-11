from torch.utils.data import Dataset
from torchvision.io import read_file, decode_image
from typing import List, Tuple
import enum
import os
import re
import torch
from pathlib import Path

class BufferType(str, enum.Enum):
    Rasterized = "Rasterized"
    SceneDepth = "SceneDepth"
    LitPrimitive = "LitPrimitive"
    GuideColored = "GuideColored"
    WorldNormal = "WorldNormal"
    Mask = "Mask"
    HighQualityRender = "HighQualityRender"
    
class FurDataset(Dataset):
    def __init__(self, base_path: Path, scenes: list[str], buffer_types: list[BufferType], stride: int = 1, transforms=None):
        self.base_path = base_path
        self.scenes = scenes
        self.buffer_types = buffer_types
        self.stride = stride
        self.indexes: List[Tuple[str, int]] = self.__get_frames_for_scenes()
        self.transforms = transforms    # (e.g. Rescale, Normalize, Random Crops, Flips) use torchvision.transforms.v2 as T

    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, idx):
        scene, frame = self.indexes[idx]
        one_channel_buffer_idx = []
        buffers = [] 
        
        for i, buffer in enumerate(self.buffer_types):
            image_tensor = self.load_and_process_buffer(buffer.value, scene, frame)
            if buffer == BufferType.SceneDepth or buffer == BufferType.Mask:
                one_channel_buffer_idx.append(i)
            buffers.append(image_tensor)

        # [C, H, W]
        target = self.load_and_process_buffer(BufferType.HighQualityRender.value, scene, frame)

        # [N_buffers + 1, C, H, W]
        all_images = torch.stack(buffers + [target], dim=0)
        

        if self.transforms:
            all_images = self.transforms(all_images)

        buffer_stack = all_images[:-1]  # [N_buffers, C, H, W]
        target = all_images[-1]         # [C, H, W]

        buffer_stack = list(buffer_stack)
        for idx in one_channel_buffer_idx:
            buffer_stack[idx] = self.__convert_to_one_channel(buffer_stack[idx]) 

        # Concatenate buffers along channel dimension
        buffer_stack = torch.cat(buffer_stack, dim=0)  # [C_total, H, W]

        # Convert to float32 and normalize for model input
        buffer_stack = buffer_stack.float() / 255.0
        target = target.float() / 255.0

        return {'bufferStack': buffer_stack, 'target': target}

    def load_and_process_buffer(self, buffer_name: str, scene: str, frame: int = 1):
        """Cache loaded buffers to avoid re-reading files"""
        PATH = self.base_path / scene / f"{buffer_name}{frame:04d}.png"
        image_data = read_file(str(PATH))
        image_tensor = decode_image(image_data)
        image_tensor = image_tensor[:3, :, :]
        return image_tensor

    def __convert_to_one_channel(self, image_tensor):
        """
        Convert a multi-channel image tensor to a single channel.
        All channels are the same since image is grayscale, so we can just take the first channel.
        """
        return image_tensor[:1]

    def __get_frames_for_scenes(self):
        """
        Find all frames for each scene filling self.index with (scene, frame) tuples
        """
        indexes = []
        frame_re = re.compile(r"HighQualityRender(\d{4})\.png$") # Regex to match highquality render files and extract frame number
        base_dir = self.base_path
        for scene in self.scenes:
            scene_dir = base_dir / scene
            if not scene_dir.is_dir():
                # skip missing scene directories
                continue
            frames = set()
            try:
                for fname in os.listdir(scene_dir):
                    m = frame_re.search(fname)
                    if m:
                        frames.add(int(m.group(1)))

                sorted_frames = sorted(list(frames))
                final_frames = sorted_frames[::self.stride]  # Apply stride to select frames

                for f in final_frames:
                    indexes.append((scene, f))
                
            except Exception:
                # if listing fails for any reason, skip this scene
                continue

        return indexes
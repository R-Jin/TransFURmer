from typing import TypedDict
try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack
import torch.nn as nn
from models.network_swinir import SwinIR


class SwinIRKwargs(TypedDict, total=False):
    img_size: int | tuple[int, int]
    patch_size: int | tuple[int, int]
    embed_dim: int
    depths: list[int]
    num_heads: list[int]
    window_size: int
    mlp_ratio: float
    qkv_bias: bool
    qk_scale: float | None
    drop_rate: float
    attn_drop_rate: float
    drop_path_rate: float
    norm_layer: type[nn.Module]
    ape: bool
    patch_norm: bool
    use_checkpoint: bool
    upscale: int
    img_range: float
    upsampler: str
    resi_connection: str


class SwinIR_out3(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int = 3,
        head_kernel_size: int = 3,
        head_padding: int = 1,
        **swinir_kwargs: Unpack[SwinIRKwargs],
    ):
        super().__init__()
        self.swinir = SwinIR(in_chans=in_chans, **swinir_kwargs)
        self.rgb_head = nn.Conv2d(in_chans, out_chans, head_kernel_size, padding=head_padding)

    def forward(self, x):
        return self.rgb_head(self.swinir(x))
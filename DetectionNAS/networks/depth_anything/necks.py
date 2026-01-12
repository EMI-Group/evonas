import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet.registry import MODELS


@MODELS.register_module()
class Feature2Pyramid(nn.Module):
    """
    Build a multi-scale pyramid from a single-scale ViT feature map.
    - Input:  x (Tensor) or [x], shape [B, embed_dim, H, W] where H=W=img/patch_size
    - Output: tuple of 5 feature maps, each [B, out_channels, Hi, Wi]

    Default rescales produce integer strides for patch_size=14:
      rescales = [2, 1, 0.5, 0.25, 0.125]
      strides  = [7,14,  28,   56,    112]  (relative to input image)
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        out_channels: int = 256,
        rescales=(2, 1, 0.5, 0.25, 0.125),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.rescales = list(rescales)

        # rescale ops
        self.ops = nn.ModuleList()
        for k in self.rescales:
            if k == 2:
                op = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 4:
                op = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    build_norm_layer(norm_cfg, embed_dim)[1],
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 1:
                op = nn.Identity()
            elif k == 0.5:
                op = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                op = nn.MaxPool2d(kernel_size=4, stride=4)
            elif k == 0.125:
                op = nn.MaxPool2d(kernel_size=8, stride=8)
            else:
                raise KeyError(f"Unsupported rescale factor: {k}")
            self.ops.append(op)

        # per-level projection to out_channels (keeps mmdet heads unchanged)
        self.proj = nn.ModuleList()
        for _ in self.rescales:
            self.proj.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1],
                    nn.GELU(),
                )
            )

    def forward(self, inputs):
        # accept Tensor or list/tuple with a single Tensor
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) >= 1
            x = inputs[-1]  # take last by default
        else:
            x = inputs

        assert x.dim() == 4, f"Expected [B,C,H,W], got {x.shape}"
        assert x.size(1) == self.embed_dim, f"Expected C={self.embed_dim}, got {x.size(1)}"

        outs = []
        for op, proj in zip(self.ops, self.proj):
            y = op(x)
            y = proj(y)
            outs.append(y)
        return tuple(outs)

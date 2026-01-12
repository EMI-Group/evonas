import torch
from mmengine.model import BaseModule
from torch import nn

from mmdet.registry import MODELS



@MODELS.register_module()
class DINOv2(nn.Module):
    """Use DINOv2 pre-trained models
    """

    def __init__(self, version='large', freeze=False, load_from=None, use_adapters=False):
        super().__init__()
        
        if version == 'large':
            self.dinov2 = torch.hub.load('DetectionNAS/networks/depth_anything/torchhub/facebookresearch_dinov2_main', 'dinov2_vitl14', source='local', pretrained=True)
        else:
            raise NotImplementedError

        if load_from is not None:
            d = torch.load(load_from, map_location='cpu')
            new_d = {}
            for key, value in d.items():
                if 'pretrained' in key:
                    new_d[key.replace('pretrained.', '')] = value
            self.dinov2.load_state_dict(new_d)
        
        self.freeze = freeze
        if self.freeze:
            for p in self.dinov2.parameters():
                p.requires_grad_(False)

        self.use_adapters = use_adapters
        if self.use_adapters:
            self.adapters = nn.ModuleList([
                MLPAdapter(1024, bottleneck=256,
                           scale=1.0, drop=0.0)
                for _ in range(2)
            ])
        else:
            self.adapters = None
        
    def forward(self, inputs):
        B, _, h, w = inputs.shape
        
        if self.freeze:
            with torch.no_grad():
                features = self.dinov2.get_intermediate_layers(inputs, 4)
        else:
            features = self.dinov2.get_intermediate_layers(inputs, 4)

        features = list(features)

        if self.use_adapters:
            # 只对最后两层做 adapter
            features[-2] = self.adapters[0](features[-2])
            features[-1] = self.adapters[1](features[-1])
        
        outs = []
        for feature in features:
            C = feature.shape[-1]
            feature = feature.permute(0, 2, 1).reshape(B, C, h // 14, w // 14).contiguous()
            outs.append(feature)

        return outs


class MLPAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64, scale: float = 1.0, drop: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.scale = scale

        # 让 adapter 初始接近 0（不破坏预训练特征）
        nn.init.zeros_(self.up.weight)

    def forward(self, x):  # x: [B, N, C]
        y = self.norm(x)
        y = self.down(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.up(y)
        return x + self.scale * y
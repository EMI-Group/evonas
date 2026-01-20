import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet.registry import MODELS
from mmdet.models.necks.fpn import FPN as MMDetFPN

@MODELS.register_module()
class DINOv2LayersThenFPN(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 out_channels=256,
                 norm_cfg=dict(type="SyncBN", requires_grad=True),
                 fpn_norm_cfg=dict(type="SyncBN", requires_grad=True),
                 num_outs=5,
                 add_extra_convs=False,
                 upsample_cfg=dict(mode="nearest")):
        super().__init__()
        self.to_pyr = MODELS.build(dict(
            type="DINOv2LayersToPyramidInputs",
            in_channels=in_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
        ))
        self.fpn = MMDetFPN(
            in_channels=[out_channels]*4,   # ✅ FPN 接 4 层
            out_channels=out_channels,
            num_outs=num_outs,              # ✅ 输出 5 层（P6由maxpool补）
            add_extra_convs=add_extra_convs,
            norm_cfg=fpn_norm_cfg,
            upsample_cfg=upsample_cfg,
        )

    def forward(self, inputs):
        # inputs 是 backbone 输出的 4 个 feature maps: [B,1024,Hp,Wp] * 4
        feats4 = self.to_pyr(inputs)
        outs5 = self.fpn(feats4)
        return outs5
    

@MODELS.register_module()
class DINOv2LayersToPyramidInputs(nn.Module):
    """输入: 4个同尺度特征 [B, C, Hp, Wp]
       输出: 4个不同尺度特征，stride=[7,14,28,56]，通道统一到 out_channels
    """
    def __init__(self, in_channels=1024, out_channels=256,
                 norm_cfg=dict(type="SyncBN", requires_grad=True)):
        super().__init__()

        # 对 4 个 level 分别做尺度变换：up2, id, down2, down4
        self.resize = nn.ModuleList([
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),  # ×2
            nn.Identity(),                                                         # ×1
            nn.MaxPool2d(kernel_size=2, stride=2),                                  # /2
            nn.MaxPool2d(kernel_size=4, stride=4),                                  # /4
        ])

        # 每层投影到 256
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.GELU(),
            )
            for _ in range(4)
        ])

    def forward(self, feats4):
        assert isinstance(feats4, (list, tuple)) and len(feats4) == 4
        outs = []
        for i in range(4):
            x = self.resize[i](feats4[i])
            x = self.proj[i](x)
            outs.append(x)
        return tuple(outs)  # 4 层：stride=[7,14,28,56]

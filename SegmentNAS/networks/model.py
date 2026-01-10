import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

# from .mambaVision import Block, MambaVisionLayer
from .uper_crf_head import PSP, StripPooling

from .SpaMamba.spatialmamba import SpatialMambaLayer

from mmseg.registry import MODELS
from mmengine.model import BaseModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmcv.cnn import ConvModule
########################################################################################################################
BN = dict(type='SyncBN', requires_grad=True)
ACT = dict(type='ReLU', inplace=True)

@MODELS.register_module()
class MambaBackbone(nn.Module):
    def __init__(self, version=None, pretrained=None, remap=False, input_height=-1, input_width=-1, 
                 width_multiplier=1.0, selected_config=None, **kwargs):
        super().__init__()

        self.version = version
        self.pretrained = pretrained

        ### encoder
        if version == 'SuperNet':
            from .VSSD.mamba2 import Backbone_VMAMBA2
            self.backbone = Backbone_VMAMBA2(
                image_size=(input_height, input_width),  # not useful
                patch_size=4,  # 无实际意义
                in_chans=3,
                embed_dim=64,
                depths=[2, 4, 8, 4],
                num_heads=[2, 4, 8, 16],
                mlp_ratio=4.0,
                drop_rate=0.0,
                drop_path_rate=0.2,
                simple_downsample=False,
                simple_patch_embed=False,
                ssd_expansion=4,
                ssd_ngroups=1,
                ssd_chunk_size=256,
                linear_attn_duality=True,
                lepe=False,
                attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
                bidirection=False,
                d_state=64,
                ssd_positve_dA=True,
                # pretrained weight
                pretrained=None,  # after
                remap=remap
            )
            in_channels = [64, 128, 256, 512]
        
        elif version == 'VSSD_final':
            from .VSSD.mamba2_final import Backbone_VMAMBA2_Final
            '''bulid one subset(not supernet) by config'''
            self.backbone = Backbone_VMAMBA2_Final(
                image_size=(input_height, input_width),
                patch_size=4,  # 无实际意义
                in_chans=3,
                embed_dim=make_divisible(64 * width_multiplier),
                depths=selected_config['depth'],
                num_heads=[2, 4, 8, 16],
                mlp_ratio=selected_config['mlp_ratio'],
                drop_rate=0.0,
                drop_path_rate=0.2,
                simple_downsample=False,
                simple_patch_embed=False,
                ssd_expansion=selected_config['ssd_expand'],
                ssd_ngroups=1,
                ssd_chunk_size=256,
                linear_attn_duality=True,
                lepe=False,
                attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
                bidirection=False,
                d_state=selected_config['d_state'],
                ssd_positve_dA=True,
                # pretrained weight
                pretrained=None  # after
            )
            in_channels = [64, 128, 256, 512]
            # scale channel
            in_channels = [make_divisible(c * width_multiplier) for c in in_channels]
            

        # self.init_weights()
        # self.backbone.load_pretrained(pretrained)
        
    def init_weights(self):
        # assert False,'test enter MambaDetection init_weight'
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
        self.backbone.load_pretrained(self.pretrained)


    def forward(self, imgs):
        feats = self.backbone(imgs)
        # for i, f in enumerate(feats):
        #     print(f'feats[{i}].shape = {f.shape}')

        return feats


@MODELS.register_module()
class UPerHead_with_mamba(BaseDecodeHead):
    def __init__(self, in_channels=[64, 128, 256, 512], **kwargs):
        super().__init__(in_channels=in_channels, input_transform='multiple_select', **kwargs)
        # PSP Module
        self.PPM = nn.Sequential(StripPooling(in_channels[3], (20,12)),
                                StripPooling(in_channels[3], (20,12)))
        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        ### FPN
        depths = [1,1,1,1]

        # self.spa_mab3 = SpatialMambaLayer(dim=in_channels[3], depth=depths[0], d_state=1)
        self.spa_mab2 = SpatialMambaLayer(dim=in_channels[2], depth=depths[1], d_state=1)
        self.spa_mab1 = SpatialMambaLayer(dim=in_channels[1], depth=depths[2], d_state=1)
        self.spa_mab0 = SpatialMambaLayer(dim=in_channels[0], depth=depths[3], d_state=1)

        self.proj_out3 = ConvModule(in_channels[3], in_channels[3] // 2, kernel_size=1, norm_cfg=BN, act_cfg=ACT)
        self.proj_out2 = ConvModule(in_channels[2], in_channels[2] // 2, kernel_size=1, norm_cfg=BN, act_cfg=ACT)
        self.proj_out1 = ConvModule(in_channels[1], in_channels[1] // 2, kernel_size=1, norm_cfg=BN, act_cfg=ACT)
        self.proj_final = ConvModule(in_channels[0], self.channels, kernel_size=3, padding=1, norm_cfg=BN, act_cfg=ACT)
            
        self.conv_p3 = ConvModule(in_channels[3], self.channels, 3, padding=1, norm_cfg=BN, act_cfg=ACT)
        self.conv_p2 = ConvModule(in_channels[2], self.channels, 3, padding=1, norm_cfg=BN, act_cfg=ACT)
        self.conv_p1 = ConvModule(in_channels[1], self.channels, 3, padding=1, norm_cfg=BN, act_cfg=ACT)

    def _upsample_to(self, x, ref):
        """bilinear 上采样到 ref 的空间尺寸"""
        return resize(
            x,
            size=ref.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

    def _forward_feature(self, feats):
        e3 = self.PPM(feats[3])

        e2 = self._upsample_to(self.proj_out3(e3), feats[2])  # 1/32 -> 1/16
        e2 = self.spa_mab2(e2 + feats[2])

        e1 = self._upsample_to(self.proj_out2(e2), feats[1])
        e1 = self.spa_mab1(e1 + feats[1])

        e0 = self._upsample_to(self.proj_out1(e1), feats[0])
        e0 = self.spa_mab0(e0 + feats[0])
        e0 = self.proj_final(e0)  #  W/4

        e1 = self.conv_p1(e1)  #  W/8
        e2 = self.conv_p2(e2)  #  W/16
        e3 = self.conv_p3(e3)  #  W/32

        # ====== 严格对齐 UPerHead 的做法：对齐到 fpn_outs[0] 的空间尺寸，然后通道拼接 ======
        fpn_outs = [e0, e1, e2, e3]             # 顺序与 UPerHead 一致：最浅层在前
        base_size = fpn_outs[0].shape[2:]       # 以最高分辨率 e0 为基准尺寸

        # 与 UPerHead 相同的循环方向（从后往前），逐个 resize 到 base_size
        for i in range(len(fpn_outs) - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=base_size,
                mode='bilinear',
                align_corners=self.align_corners
            )

        # 在通道维拼接（B, 4*C, H/4, W/4）
        fpn_outs = torch.cat(fpn_outs, dim=1)

        # 3x3 bottleneck 压回 self.channels（B, C, H/4, W/4）
        feats_out = self.fpn_bottleneck(fpn_outs)

        return feats_out  # 与 UPerHead 一样返回单张融合特征图

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        # _forward_feature 输出单个特征图：shape=(2, 512, 128, 256), dtype=torch.float32, device=cuda:0
        output = self.cls_seg(output)
        return output
        


def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)
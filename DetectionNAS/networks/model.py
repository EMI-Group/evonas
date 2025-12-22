import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

# from .mambaVision import Block, MambaVisionLayer
from .uper_crf_head import PSP, StripPooling

from .SpaMamba.spatialmamba import SpatialMambaLayer

from mmdet.registry import MODELS
from mmengine.model import BaseModule
########################################################################################################################

@MODELS.register_module()
class MambaDetection(nn.Module):
    """
    network based on VSSD-T + SpatialMamba.
    """
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

        elif version == 'VSSD':
            from .VSSD.mamba2_fixed import Backbone_VMAMBA2_Fixed
            '''fixed arch'''
            self.backbone = Backbone_VMAMBA2_Fixed(
                image_size=(input_height, input_width),
                patch_size=4,  # 无实际意义
                in_chans=3,
                embed_dim=64,
                depths=[2, 4, 8, 4],  # note
                num_heads=[2, 4, 8, 16],
                mlp_ratio=4.0,  # note
                drop_rate=0.0,
                drop_path_rate=0.2,
                simple_downsample=False,
                simple_patch_embed=False,
                ssd_expansion=2,  # note
                ssd_ngroups=1,
                ssd_chunk_size=256,
                linear_attn_duality=True,
                lepe=False,
                attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
                bidirection=False,
                d_state=64,  # note
                ssd_positve_dA=True,
                # pretrained weight
                pretrained=None  # after
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
class MambaFPN(nn.Module):
    def __init__(self, in_channels, out_channels, neck_type='sp', **kwargs):
        super().__init__()

        ### decoder
        depths = [1,1,1,1]

        self.spa_mab3 = SpatialMambaLayer(dim=in_channels[3], depth=depths[0], d_state=1, mlp_ratio=4.0)
        self.spa_mab2 = SpatialMambaLayer(dim=in_channels[2], depth=depths[1], d_state=1, mlp_ratio=4.0)
        self.spa_mab1 = SpatialMambaLayer(dim=in_channels[1], depth=depths[2], d_state=1, mlp_ratio=4.0)
        self.spa_mab0 = SpatialMambaLayer(dim=in_channels[0], depth=depths[3], d_state=1, mlp_ratio=4.0)

        self.proj_out3 = nn.Conv2d(in_channels[3], in_channels[3]*2, kernel_size=3, stride=1, padding=1)
        self.proj_out2 = nn.Conv2d(in_channels[2], in_channels[2]*2, kernel_size=3, stride=1, padding=1)
        self.proj_out1 = nn.Conv2d(in_channels[1], in_channels[1]*2, kernel_size=3, stride=1, padding=1)
        self.proj_final = nn.Conv2d(in_channels[0], out_channels, kernel_size=3, stride=1, padding=0)
            
        self.conv_p3 = nn.Conv2d(in_channels[3]*2, out_channels, kernel_size=1)
        self.conv_p2 = nn.Conv2d(in_channels[2]*2, out_channels, kernel_size=1)
        self.conv_p1 = nn.Conv2d(in_channels[1]*2, out_channels, kernel_size=1)

        self.max_pool = nn.MaxPool2d(kernel_size=1, stride=2)
    

    def forward(self, feats):
        e3 = self.spa_mab3(feats[3])
        e3 = self.proj_out3(e3)

        e2 = self.spa_mab2(nn.PixelShuffle(2)(e3))
        e2 = e2 + feats[2]
        e2 = self.proj_out2(e2)

        e1 = self.spa_mab1(nn.PixelShuffle(2)(e2))
        e1 = e1 + feats[1]
        e1 = self.proj_out1(e1)

        e0 = self.spa_mab0(nn.PixelShuffle(2)(e1))
        e0 = e0 + feats[0]
        e0 = self.proj_final(e0)  #  W/4

        e1 = self.conv_p1(e1)  #  W/8
        e2 = self.conv_p2(e2)  #  W/16
        e3 = self.conv_p3(e3)  #  W/32
        ee = self.max_pool(e3)   #  W/64

        out = (e0, e1, e2, e3, ee)

        return out


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


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
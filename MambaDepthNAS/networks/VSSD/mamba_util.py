# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Demystify Mamba in Vision: A Linear Attention Perspective
# Modified by Dongchen Han
# -----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# yzh
from .module.Linear_super import LinearSuper
import re
from typing import Dict, List

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MlpSuper(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LinearSuper(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = LinearSuper(hidden_features, self.out_features)
        self.drop = nn.Dropout(drop)

    def set_sample_config(self, in_features, sample_mlp_ratio):
        self.fc1.set_sample_config(in_features, int(in_features*sample_mlp_ratio))
        self.fc2.set_sample_config(int(in_features*sample_mlp_ratio), self.out_features)

    def forward(self, x):
        # print('before fc1:', x.shape)
        x = self.fc1(x)
        # print('after fc1:', x.shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # print('after fc2:', x.shape)
        x = self.drop(x)
        return x
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SimpleStem(nn.Module):
    r'''
    Simple Stem

    '''

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.conv1 = ConvLayer(in_chans, embed_dim, kernel_size=4, stride=4, padding=0, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # padding

        x = self.norm(self.conv1(x).flatten(2).transpose(1, 2))
        return x



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(self, input_resolution, dim, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x, H=None, W=None):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        if H & W is None:
            H, W = self.input_resolution
            assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        return x

class SimplePatchMerging(nn.Module):
    r""" Simple Patch Merging Layer.

        Args:
            input_resolution (tuple[int]): Resolution of input feature.
            dim (int): Number of input channels.
        """

    def __init__(self, input_resolution, dim, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels), kernel_size=3,stride=2, padding=1, norm=None),
        )
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, x, H=None, W=None):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        if H & W is None:
            H, W = self.input_resolution
            assert L == H * W, "input feature has wrong size"
        #assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        x = self.norm(x)
        return x


def expand_depth(ori_state_dict, new_state_dict):
    """
    Expand the depth of the state_dict by duplicating the last layer.
    
    Args:
        ori_state_dict (dict): The original state_dict with keys like 'layers.0.blocks.1.mlp.fc1.weight'.
        new_state_dict (dict): The new model's state_dict.
    
    Returns:
        dict: The modified state_dict with expanded depth.
    """
    missing_keys = []

    for key in new_state_dict.keys():
        if key in ori_state_dict and ori_state_dict[key].shape == new_state_dict[key].shape:
            new_state_dict[key] = ori_state_dict[key].clone()
        else:
            match = re.match(r'(layers\.\d+\.blocks\.)(\d+)(\..+)', key)
            if match:
                stage, layer, suffix = match.groups()
                layer_index = int(layer)

                while layer_index >= 0:
                    src_key = f"{stage}{layer_index}{suffix}"
                    if src_key in ori_state_dict and ori_state_dict[src_key].shape == new_state_dict[key].shape:
                        new_state_dict[key] = ori_state_dict[src_key].clone()
                        # print(f"Fallback copy: {src_key} -> {key}")
                        break
                    layer_index -= 1
                else:
                    missing_keys.append(key)
            else:
                missing_keys.append(key)
                
    print('missing_keys=', missing_keys)
    # 'outnorm0.weight', 'outnorm0.bias', 'outnorm1.weight', 'outnorm1.bias', 'outnorm2.weight', 'outnorm2.bias', 'outnorm3.weight', 'outnorm3.bias' is for classification task

    return new_state_dict


def select_depth_from_supernet(supernet_state_dict: Dict[str, torch.Tensor],
                                target_state_dict: Dict[str, torch.Tensor],
                                depth_mask: List[List[int]]) -> Dict[str, torch.Tensor]:
    """
    Copy selected blocks from a supernet's state_dict to the target model's state_dict
    according to a binary depth mask.

    Args:
        supernet_state_dict: The full supernet parameter dictionary.
        target_state_dict: The new model's parameter dictionary (to update).
        depth_mask: List of lists indicating which blocks to keep per stage.

    Returns:
        Updated target_state_dict with selected weights copied from the supernet.
    """
    missing_keys = []

    # Precompute mapping from new block index -> supernet block index per stage
    block_index_map = []
    for stage_mask in depth_mask:  # [[1, 1], [0, 0, 0, 1], ...]
        stage_map = []
        for i, m in enumerate(stage_mask):  # [1, 1]
            if m == 1:
                stage_map.append(i)  # [0, 1]
        block_index_map.append(stage_map)  # [[0, 1], [3], ...]

    for key in target_state_dict.keys():
        match = re.search(r'layers\.(\d+)\.blocks\.(\d+)(\..+)', key)
        if match:
            stage_str, block_str, suffix = match.groups()
            stage = int(stage_str)
            block = int(block_str)
            try:
                mapped_block = block_index_map[stage][block]
                super_key = re.sub(rf'(layers\.{stage}\.blocks\.){block}', rf'\g<1>{mapped_block}', key)

                if super_key in supernet_state_dict and supernet_state_dict[super_key].shape == target_state_dict[key].shape:
                    target_state_dict[key] = supernet_state_dict[super_key].clone()
                    print(f'{super_key} --> {key}')
                else:
                    missing_keys.append(key)
                    print(f'1 {key}, {supernet_state_dict[super_key].shape}, {target_state_dict[key].shape}')
            except (IndexError, KeyError):
                missing_keys.append(key)
                print(f'2 {key}')
        else:
            # direct copy if key exists
            if key in supernet_state_dict and supernet_state_dict[key].shape == target_state_dict[key].shape:
                target_state_dict[key] = supernet_state_dict[key].clone()
            else:
                missing_keys.append(key)
                print(f'3 {key}')

    print("Missing keys:", missing_keys)
    return target_state_dict

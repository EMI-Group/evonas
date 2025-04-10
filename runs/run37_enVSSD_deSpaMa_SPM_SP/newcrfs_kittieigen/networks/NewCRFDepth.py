import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import Block, MambaVisionLayer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP, StripPooling, SPBlock

from .SpaMamba.spatialmamba import SpatialMambaLayer
########################################################################################################################
'''
encoder-decoder = VSSD-SpatialMamba
load pretrained weight (on imagenet-1k)
PPM ==> SPM + StripPooling
'''

class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture.
    """
    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.version = version
        self.with_neck = False
        # print('max_depth',max_depth)
        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        ### encoder
        if version == 'VSSD':
            from .VSSD.mamba2 import Backbone_VMAMBA2
            self.backbone = Backbone_VMAMBA2(
                image_size=(352,1120),  # 需配合数据集
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
                ssd_expansion=2,
                ssd_ngroups=1,
                ssd_chunk_size=256,
                linear_attn_duality=True,
                lepe=False,
                attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
                bidirection=False,
                d_state=64,
                ssd_positve_dA=True,
                # pretrained weight
                pretrained=pretrained
            )
            in_channels = [64, 128, 256, 512]

        elif version == 'MambaVision':
            from .mamba import Block, MambaVision
            model_path = './mambavision_tiny_1k.pth.tar'
            depths = [1, 3, 8, 4]
            num_heads = [2, 4, 8, 16]
            window_size = [8, 8, 14, 7]
            dim = 80
            in_dim = 32
            mlp_ratio = 4
            drop_path_rate = 0.2

            self.backbone = MambaVision(depths=depths,
                                        num_heads=num_heads,
                                        window_size=window_size,
                                        dim=dim,
                                        in_dim=in_dim,
                                        mlp_ratio=mlp_ratio,
                                        drop_path_rate=drop_path_rate)
            if pretrained:
                self.backbone._load_state_dict(model_path)
            
            in_channels = [80, 160, 320, 640]

        elif version == 'MLLA':
            from .mlla import MLLA, load_pretrained
            self.backbone = MLLA(img_size=(352,1216),
                                patch_size=4,
                                in_chans=3,
                                embed_dim=64,
                                depths=[2, 4, 8, 4],
                                num_heads=[2, 4, 8, 16],
                                mlp_ratio=4,
                                qkv_bias=True,
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                use_checkpoint=False)
            ckpt_path = './MLLA-T.pth'
            load_pretrained(ckpt_path, self.backbone, skip_RoPE=True)
            in_channels = [64, 128, 256, 512]

        else:
            from .swin_transformer import SwinTransformer
            window_size = int(version[-2:])

            if version[:-2] == 'base':
                embed_dim = 128
                depths = [2, 2, 18, 2]
                num_heads = [4, 8, 16, 32]
                in_channels = [128, 256, 512, 1024]
            elif version[:-2] == 'large':
                embed_dim = 192
                depths = [2, 2, 18, 2]
                num_heads = [6, 12, 24, 48]
                in_channels = [192, 384, 768, 1536]
            elif version[:-2] == 'tiny':
                embed_dim = 96
                depths = [2, 2, 6, 2]
                num_heads = [3, 6, 12, 24]
                in_channels = [96, 192, 384, 768]

            backbone_cfg = dict(
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                use_checkpoint=False,
                frozen_stages=frozen_stages
            )
            self.backbone = SwinTransformer(**backbone_cfg)

        embed_dim = 512
        # decoder_cfg = dict(
        #     in_channels=in_channels,
        #     in_index=[0, 1, 2, 3],
        #     pool_scales=(1, 2, 3, 6),
        #     channels=embed_dim,
        #     dropout_ratio=0.0,
        #     num_classes=32,
        #     norm_cfg=norm_cfg,
        #     align_corners=False
        # )

        ### decoder
        win = 7
        crf_dims = [64, 128, 256, 512]
        v_dims = [32, 64, 128, embed_dim]

        # self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)  # x的输入维度，整体输出维度，窗口大小，v的输入维度，注意力头数
        # self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        # self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        # self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        depths = [1,1,1,1]

        self.spa_mab3 = SpatialMambaLayer(dim=in_channels[3], depth=depths[0], d_state=1, mlp_ratio=4.0)
        self.spa_mab2 = SpatialMambaLayer(dim=in_channels[2], depth=depths[1], d_state=1, mlp_ratio=4.0)
        self.spa_mab1 = SpatialMambaLayer(dim=in_channels[1], depth=depths[2], d_state=1, mlp_ratio=4.0)
        self.spa_mab0 = SpatialMambaLayer(dim=in_channels[0], depth=depths[3], d_state=1, mlp_ratio=4.0)

        self.proj_out3 = nn.Conv2d(in_channels[3], in_channels[3]*2, 3, 1, 1)
        self.proj_out2 = nn.Conv2d(in_channels[2], in_channels[2]*2, 3, 1, 1)
        self.proj_out1 = nn.Conv2d(in_channels[1], in_channels[1]*2, 3, 1, 1)
        self.proj_final = nn.Conv2d(in_channels[0], crf_dims[0], 3, 1, 1)
        
        # self.decoder = PSP(**decoder_cfg)  # 影响一个点
        self.PPM = nn.Sequential(SPBlock(in_channels[3],in_channels[3]),
                                    StripPooling(in_channels[3], (20,12)),
                                     StripPooling(in_channels[3], (20,12)))
        self.disp_head1 = DispHead(input_dim=crf_dims[0])

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(crf_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))

        self.min_depth = min_depth
        self.max_depth = max_depth

    #     self.init_weights(pretrained=pretrained)

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone and heads.
    #     # 检查了,无实际意义
    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """
    #     self.decoder.init_weights()
        

    def upsample_mask(self, disp, mask):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, 4*H, 4*W)

    def forward(self, imgs):
        feats = self.backbone(imgs)

        # for ii in range(len(feats)):
        #     print(f'feat[{ii}]: {feats[ii].shape}')
        # assert False

        ppm_out = self.PPM(feats[3])

        e3 = self.spa_mab3(ppm_out)
        e3 = e3 + feats[3]
        e3 = nn.PixelShuffle(2)(self.proj_out3(e3))

        e2 = self.spa_mab2(e3)
        e2 = e2 + feats[2]
        e2 = nn.PixelShuffle(2)(self.proj_out2(e2))

        e1 = self.spa_mab1(e2)
        e1 = e1 + feats[1]
        e1 = nn.PixelShuffle(2)(self.proj_out1(e1))

        e0 = self.spa_mab0(e1)
        e0 = e0 + feats[0]
        e0 = self.proj_final(e0)

        d1 = self.disp_head1(e0, 4)

        depth = d1 * self.max_depth

        return depth



class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


class DispUnpack(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128):
        super(DispUnpack, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x, output_size):
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x)) # [b, 16, h/4, w/4]
        # x = torch.reshape(x, [x.shape[0], 1, x.shape[2]*4, x.shape[3]*4])
        x = self.pixel_shuffle(x)

        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
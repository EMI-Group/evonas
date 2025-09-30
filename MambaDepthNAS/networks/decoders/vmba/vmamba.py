import torch
import torch.nn as nn
import torch.nn.functional as F

from .VMamba.vmamba import VSSBlock, VSSM

from collections import OrderedDict

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
def make_Mamba_layer(
        dim=96, 
        drop_path=[0.0], 
        use_checkpoint=False, 
        norm_layer=LayerNorm2d,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        **kwargs,
    ):
    # if channel first, then Norm and Output are both channel_first
    depth = len(drop_path)
    blocks = []
    for d in range(depth):
        blocks.append(VSSBlock(
            hidden_dim=dim, 
            drop_path=drop_path[d],
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type=forward_type,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            gmlp=gmlp,
            use_checkpoint=use_checkpoint,
        ))
    
    return nn.Sequential(OrderedDict(
        blocks=nn.Sequential(*blocks,),
        downsample=downsample,
    ))


    

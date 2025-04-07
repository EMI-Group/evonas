import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

bs = 8
resolution = 640

'''Swin-T'''  # Throughput 202
# from networks.swin_transformer import SwinTransformer
# backbone_cfg = dict(
#     embed_dim=96,
#     depths=[2, 2, 6, 2],
#     num_heads=[3, 6, 12, 24],
#     window_size=7,
#     ape=False,
#     drop_path_rate=0.3,
#     patch_norm=True,
#     use_checkpoint=False,
#     frozen_stages=-1
# )
# model = SwinTransformer(**backbone_cfg)

'''MambaVsion-T'''  # Throughput 346 (Group+CAM); Throughput 348 (Orginal)
# from networks.mamba import MambaVision
# model = MambaVision(depths=[1, 3, 8, 4],
#                     num_heads=[2, 4, 8, 16],
#                     window_size=[8, 8, 14, 7],
#                     dim=80,
#                     in_dim=32,
#                     mlp_ratio=4,
#                     drop_path_rate=0.2)

'''MLLA-T'''  # Throughput 159
# from networks.mlla import MLLA
# model = MLLA(img_size=(640,640),
#             patch_size=4,
#             in_chans=3,
#             embed_dim=64,
#             depths=[2, 4, 8, 4],
#             num_heads=[2, 4, 8, 16],
#             mlp_ratio=4,
#             qkv_bias=True,
#             drop_rate=0.0,
#             drop_path_rate=0.2,
#             ape=False,
#             use_checkpoint=False)

'''VSSD-T'''  # Throughput 124
from networks.VSSD.mamba2 import Backbone_VMAMBA2
model = Backbone_VMAMBA2(
    image_size=(640,640),
    patch_size=4,
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
)

input_data = torch.randn((bs, 3, resolution, resolution), device='cuda').cuda()
input_data = input_data.to(memory_format=torch.channels_last)
model = model.to(memory_format=torch.channels_last)
model.cuda()
model.eval()
   
# warm up
with torch.cuda.amp.autocast():
    for ii in range(100):
        with torch.no_grad():
            output = model(input_data)

# speed
import time
import numpy as np

timer = []
start_time = time.time()
runs=500
with torch.cuda.amp.autocast(True):
    for ii in range(runs):
        start_time_loc = time.time()
        with torch.no_grad():
            output = model(input_data)

        timer.append(time.time()-start_time_loc)
    torch.cuda.synchronize()
end_time = time.time()

print(f"Throughput {int(bs * 1.0 / ((np.median(timer))))}")
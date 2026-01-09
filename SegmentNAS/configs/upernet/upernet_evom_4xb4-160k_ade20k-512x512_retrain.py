_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MambaBackbone',
        version='SuperNet',
        input_height=512,
        input_width=1024,
        width_multiplier=1.0,
        pretrained='./vssd_supernet_imagenet_1k.pth'
    ),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150,act_cfg=dict(type='ReLU', inplace=False)),
    auxiliary_head=dict(in_channels=256, num_classes=150,act_cfg=dict(type='ReLU', inplace=False),)
)

train_dataloader = dict(batch_size=4) # as gpus=4


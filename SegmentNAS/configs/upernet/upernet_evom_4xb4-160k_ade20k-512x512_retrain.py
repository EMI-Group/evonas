_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        type='MambaBackbone',
        version='SuperNet',
        input_height=512,
        input_width=1024,
        width_multiplier=1.0,
        pretrained='./vssd_supernet_imagenet_1k.pth'
    ),
    decode_head=dict(
        _delete_=True,
        type='UPerHead_with_mamba',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(in_channels=256, num_classes=150,act_cfg=dict(type='ReLU', inplace=False),)
)

train_dataloader = dict(batch_size=4) # as gpus=4


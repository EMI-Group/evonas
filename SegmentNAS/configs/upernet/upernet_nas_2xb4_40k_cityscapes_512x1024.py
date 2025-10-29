_base_ = [
    '_base_/models/upernet_r50.py', '_base_/datasets/cityscapes.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MambaBackbone',
        version='SuperNet',
        input_height=512,
        input_width=1024,
        width_multiplier=1.0,
        pretrained='./vssd_supernet_imagenet_1k.pth'),
    decode_head=dict(
        _delete_=True,
        type='UPerHead_with_mamba',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        in_channels=256,
        ),
    )
train_dataloader = dict(batch_size=4)
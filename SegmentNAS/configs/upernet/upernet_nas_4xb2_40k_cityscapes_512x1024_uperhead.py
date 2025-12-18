_base_ = [
    '_base_/models/upernet_r50.py', '_base_/datasets/cityscapes.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_40k.py'
]
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
        in_channels=[64, 128, 256, 512],
        channels=256,
        ),
    auxiliary_head=dict(
        in_channels=256,
        ),
    test_cfg=dict(mode='slide', crop_size=(512, 1024), stride=(341, 682))
    )
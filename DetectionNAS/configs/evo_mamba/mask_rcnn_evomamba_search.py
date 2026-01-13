_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance_only_bbox.py',  # subset of COCO for fast validation
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
# optimizer
model = dict(
    backbone=dict(
        type='MambaDetection',
        version='SuperNet',
        input_height=800,
        input_width=1333,
        width_multiplier=1.0,
        pretrained='./vssd_supernet_imagenet_1k.pth'),
    neck=dict(
        type='MambaFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256),
    )

_base_ = [
    '_base_/datasets/coco_instance.py',
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/schedules/schedule_1x.py'
]

model = dict(
    backbone=dict(
        depth=101,
        ))
_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

custom_imports = dict(
    imports=[
        'networks.depth_anything.dinov2',
        'networks.depth_anything.necks',
    ],
    allow_failed_imports=False
)

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=14,
    ),
    backbone=dict(
        _delete_=True,
        type='DINOv2',
        version='large',
        freeze=True,   # 冻结 DINOv2 参数
        use_adapters=False,  # 使用 adapter 微调
        unfreeze_last_blocks=False,
    ),
    neck=dict(
        _delete_=True,
        type='DINOv2LayersThenFPN',
        in_channels=1024,
        out_channels=256,
        num_outs=5,
        add_extra_convs=False,
    ),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            # ✅ scales要跟着 stride 变：原版 stride=4 配 scale=8 => base=32
            # 现在最细层 stride=7，为了接近 32，scale 选 4（7*4=28）更合理
            scales=[4],
            ratios=[0.5, 1.0, 2.0],
            strides=[7, 14, 28, 56, 112],
        ),
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[7, 14, 28, 56],
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[7, 14, 28, 56],
        ),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=80,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
        ),
    ),
)

find_unused_parameters=True

optim_wrapper = dict(type='AmpOptimWrapper')

train_dataloader = dict(
    batch_size=2,)

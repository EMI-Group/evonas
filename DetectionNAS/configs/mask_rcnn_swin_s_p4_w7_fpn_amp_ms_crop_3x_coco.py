_base_ = './mask_rcnn_swin_t_p4_w7_fpn_ms_crop_3x_coco.py'
optim_wrapper = dict(type='AmpOptimWrapper')
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
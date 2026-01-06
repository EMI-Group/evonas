#!/bin/bash
set -e

base_configs="DetectionNAS/configs/supernet_train_base_coco.txt"
out_dir="./runs/coco/00_train/FT_supernet_coco"

sleep 3
# 07 depth (last)
python DetectionNAS/train_sandw.py "@${base_configs}" \
        --ckpt_path ${out_dir}/06_ssd_2/best_bbox_mAP.pth \
        --log_directory ${out_dir}/07_depth_sandw/ \
        --num_epochs 12 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --ssd_expand "[0.5, 1, 2, 3, 4]" \
        --open_depth

echo "supernet training is fine"
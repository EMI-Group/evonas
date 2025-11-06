#!/bin/bash
set -e

base_configs="SegmentNAS/configs/upernet/00_train/supernet_train_base_cityscapes.txt"
out_dir="./runs/cityscapes/00_train/FT_supernet_cityscapes"

# 00 maxnet
python SegmentNAS/train.py "@${base_configs}" \
        --log_directory ${out_dir}/00_maxnet/  \
        --num_epochs 108

sleep 3
# 01 d_state part1
python SegmentNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/00_maxnet/best_bbox_mAP.pth \
        --log_directory ${out_dir}/01_state_1/ \
        --num_epochs 27 \
        --d_state "[48, 64]"

sleep 3
# 02 d_state part2
python SegmentNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/01_state_1/best_bbox_mAP.pth \
        --log_directory ${out_dir}/02_state_2/ \
        --num_epochs 108 \
        --d_state "[16, 32, 48, 64]"

sleep 3
# 03 mlp_ratio part1
python SegmentNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/02_state_2/best_bbox_mAP.pth \
        --log_directory ${out_dir}/03_mlp_1/ \
        --num_epochs 27 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[3.0, 3.5, 4.0]" \
        --dynamic_batch_size 2

sleep 3
# 04 mlp_ratio part2
python SegmentNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/03_mlp_1/best_bbox_mAP.pth \
        --log_directory ${out_dir}/04_mlp_2/ \
        --num_epochs 108 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --dynamic_batch_size 2

sleep 3
# 05 ssd_expand part1
python SegmentNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/04_mlp_2/best_bbox_mAP.pth \
        --log_directory ${out_dir}/05_ssd_1/ \
        --num_epochs 27 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --ssd_expand "[2, 3, 4]" \
        --dynamic_batch_size 4

sleep 3
# 06 ssd_expand part2
python SegmentNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/05_ssd_1/best_bbox_mAP.pth \
        --log_directory ${out_dir}/06_ssd_2/ \
        --num_epochs 108 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --ssd_expand "[0.5, 1, 2, 3, 4]" \
        --dynamic_batch_size 4

sleep 3
# 07 depth (last)
python SegmentNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/06_ssd_2/best_bbox_mAP.pth \
        --log_directory ${out_dir}/07_depth/ \
        --num_epochs 108 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --ssd_expand "[0.5, 1, 2, 3, 4]" \
        --dynamic_batch_size 4 \
        --open_depth

echo "supernet training is fine"
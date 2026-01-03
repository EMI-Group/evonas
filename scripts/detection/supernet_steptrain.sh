#!/bin/bash
set -e

base_configs="DetectionNAS/configs/supernet_train_base_coco.txt"
out_dir="./runs/coco/00_train/FT_supernet_coco"

# # 00 maxnet
# python DetectionNAS/train.py "@${base_configs}" \
#         --log_directory ${out_dir}/00_maxnet/  \
#         --num_epochs 12 \
#         --weight_decay 0.03

# sleep 3
# # 01 d_state part1
# python DetectionNAS/train.py "@${base_configs}" \
#         --ckpt_path ${out_dir}/00_maxnet/best_bbox_mAP.pth \
#         --log_directory ${out_dir}/01_state_1/ \
#         --num_epochs 2 \
#         --weight_decay 0.02 \
#         --d_state "[48, 64]"

sleep 3
# 02 d_state part2
python DetectionNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/01_state_1/best_bbox_mAP.pth \
        --log_directory ${out_dir}/02_state_2/ \
        --num_epochs 8 \
        --d_state "[16, 32, 48, 64]"

sleep 3
# 03 mlp_ratio part1
python DetectionNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/02_state_2/best_bbox_mAP.pth \
        --log_directory ${out_dir}/03_mlp_1/ \
        --num_epochs 2 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[3.0, 3.5, 4.0]" \
        --dynamic_batch_size 2

sleep 3
# 04 mlp_ratio part2
python DetectionNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/03_mlp_1/best_bbox_mAP.pth \
        --log_directory ${out_dir}/04_mlp_2/ \
        --num_epochs 8 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --dynamic_batch_size 2

sleep 3
# 05 ssd_expand part1
python DetectionNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/04_mlp_2/best_bbox_mAP.pth \
        --log_directory ${out_dir}/05_ssd_1/ \
        --num_epochs 2 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --ssd_expand "[2, 3, 4]" \
        --dynamic_batch_size 4

sleep 3
# 06 ssd_expand part2
python DetectionNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/05_ssd_1/best_bbox_mAP.pth \
        --log_directory ${out_dir}/06_ssd_2/ \
        --num_epochs 8 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --ssd_expand "[0.5, 1, 2, 3, 4]" \
        --dynamic_batch_size 4

sleep 3
# 07 depth (last)
python DetectionNAS/train.py "@${base_configs}" \
        --ckpt_path ${out_dir}/06_ssd_2/best_bbox_mAP.pth \
        --log_directory ${out_dir}/07_depth/ \
        --num_epochs 12 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --ssd_expand "[0.5, 1, 2, 3, 4]" \
        --dynamic_batch_size 4 \
        --open_depth

echo "supernet training is fine"
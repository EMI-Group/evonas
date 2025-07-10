#!/bin/bash
set -e

base_configs="configs/prog_shrink/base_kitti.txt"

# # 00 maxnet
# python MambaDepthNAS/train.py "@${base_configs}" \
#         --pretrain ./vssd_supernet_imagenet_1k.pth \
#         --log_directory ./runs/fine_tune_supernet_kitti/00_maxnet/  \
#         --num_epochs 40 \
#         --warmup_epochs 3

sleep 3
# 01 d_state part1
python MambaDepthNAS/train.py "@${base_configs}" \
        --ckpt_path ./runs/fine_tune_supernet_kitti/00_maxnet/abs_rel_best_weight.pth \
        --log_directory ./runs/fine_tune_supernet_kitti/01_state_1/ \
        --num_epochs 5 \
        --warmup_epochs 0 \
        --d_state "[48, 64]"

sleep 3
# 02 d_state part2
python MambaDepthNAS/train.py "@${base_configs}" \
        --ckpt_path ./runs/fine_tune_supernet_kitti/01_state_1/abs_rel_best_weight.pth \
        --log_directory ./runs/fine_tune_supernet_kitti/02_state_2/ \
        --num_epochs 20 \
        --warmup_epochs 1 \
        --d_state "[16, 32, 48, 64]"

sleep 3
# 03 mlp_ratio part1
python MambaDepthNAS/train.py "@${base_configs}" \
        --ckpt_path ./runs/fine_tune_supernet_kitti/02_state_2/abs_rel_best_weight.pth \
        --log_directory ./runs/fine_tune_supernet_kitti/03_mlp_1/ \
        --num_epochs 5 \
        --warmup_epochs 0 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[3.0, 3.5, 4.0]" \
        --dynamic_batch_size 2

sleep 3
# 04 mlp_ratio part2
python MambaDepthNAS/train.py "@${base_configs}" \
        --ckpt_path ./runs/fine_tune_supernet_kitti/03_mlp_1/abs_rel_best_weight.pth \
        --log_directory ./runs/fine_tune_supernet_kitti/04_mlp_2/ \
        --num_epochs 20 \
        --warmup_epochs 1 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --dynamic_batch_size 2

sleep 3
# 05 ssd_expand part1
python MambaDepthNAS/train.py "@${base_configs}" \
        --ckpt_path ./runs/fine_tune_supernet_kitti/04_mlp_2/abs_rel_best_weight.pth \
        --log_directory ./runs/fine_tune_supernet_kitti/05_ssd_1/ \
        --num_epochs 5 \
        --warmup_epochs 0 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --ssd_expand "[2, 3, 4]" \
        --dynamic_batch_size 4

sleep 3
# 06 ssd_expand part2
python MambaDepthNAS/train.py "@${base_configs}" \
        --ckpt_path ./runs/fine_tune_supernet_kitti/05_ssd_1/abs_rel_best_weight.pth \
        --log_directory ./runs/fine_tune_supernet_kitti/06_ssd_2/ \
        --num_epochs 20 \
        --warmup_epochs 1 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --ssd_expand "[0.5, 1, 2, 3, 4]" \
        --dynamic_batch_size 4

sleep 3
# 07 depth (last)
python MambaDepthNAS/train.py "@${base_configs}" \
        --ckpt_path ./runs/fine_tune_supernet_kitti/06_ssd_2/abs_rel_best_weight.pth \
        --log_directory ./runs/fine_tune_supernet_kitti/07_depth/ \
        --num_epochs 25 \
        --warmup_epochs 1 \
        --d_state "[16, 32, 48, 64]" \
        --mlp_ratio "[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]" \
        --ssd_expand "[0.5, 1, 2, 3, 4]" \
        --dynamic_batch_size 4 \
        --open_depth

echo "this is fine"
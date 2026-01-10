#!/bin/bash
set -e

base_configs="SegmentNAS/configs/00_train/supernet_train_base_ade20k.txt"
out_dir="./runs/ade20k/00_train/FT_supernet_ade20k"

# 00 maxnet
python SegmentNAS/train.py "@${base_configs}" \
        --log_directory ${out_dir}/00_maxnet/  \
        --total_iters 160000 \
        --eval_interval 1
#!/bin/bash

base_configs="configs/retrain_nyu.txt"
device="4,5,6,7"
num_epochs=10
warmup_epochs=1

# small
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_nyu/run0_searched_S/  \
        --num_epochs ${num_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --device ${device} \
        --mlp_ratio "[2.0, 3.0, 3.0, 0.5]" \
        --d_state "[16, 32, 32]" \
        --ssd_expand "[2, 4, 4]" \
        --depth "[[0, 1], [1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1]]"

sleep 3
# base
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_nyu/run0_searched_B/  \
        --num_epochs ${num_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --device ${device} \
        --mlp_ratio "[2.0, 0.5, 3.0, 1.0]" \
        --d_state "[16, 32, 64]" \
        --ssd_expand "[2, 4, 4]" \
        --depth "[[0, 1], [1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1]]"

sleep 3
# tiny
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_nyu/run0_searched_T/  \
        --num_epochs ${num_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --device ${device} \
        --mlp_ratio "[0.5, 0.5, 1.0, 2.0]" \
        --d_state "[16, 16, 64]" \
        --ssd_expand "[1, 1, 3]" \
        --depth "[[1, 0], [0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1]]"
        
echo "retrain is fine"
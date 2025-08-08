#!/bin/bash

base_configs="configs/retrain_nyu.txt"
device="4,5,6,7"
num_epochs=10
warmup_epochs=1


# base
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_nyu/run1_searched_B/  \
        --num_epochs ${num_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --device ${device} \
        --mlp_ratio "[2.0, 3.5, 3.0, 3.0]" \
        --d_state "[64, 48, 64]" \
        --ssd_expand "[2, 3, 4]" \
        --depth "[[0, 1], [1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 0]]"

sleep 3
# base
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_nyu/run2_searched_B/  \
        --num_epochs ${num_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --device ${device} \
        --mlp_ratio "[4.0, 3.5, 3.5, 1.0]" \
        --d_state "[48, 48, 64]" \
        --ssd_expand "[1, 4, 4]" \
        --depth "[[0, 1], [1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 1]]"

echo "retrain is fine"
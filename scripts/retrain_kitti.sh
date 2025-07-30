#!/bin/bash

base_configs="configs/retrain_kitti.txt"
device="0,1,2,3"
num_epochs=10
warmup_epochs=1

# small
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_kitti/run1_searched_S/  \
        --num_epochs ${num_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --device ${device} \
        --mlp_ratio "[4.0, 3.0, 3.0, 4.0]" \
        --d_state "[64, 16, 32]" \
        --ssd_expand "[0.5, 4, 2]" \
        --depth "[[1, 0], [0, 1, 0, 0], [1, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1]]"

sleep 3
# base
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_kitti/run1_searched_B/  \
        --num_epochs ${num_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --device ${device} \
        --mlp_ratio "[3.5, 4.0, 3.0, 3.0]" \
        --d_state "[64, 16, 32]" \
        --ssd_expand "[0.5, 4, 4]" \
        --depth "[[1, 0], [0, 1, 1, 0], [1, 1, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1]]"

sleep 3
# tiny
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_kitti/run1_searched_T/  \
        --num_epochs ${num_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --device ${device} \
        --mlp_ratio "[3.0, 1.0, 2.0, 0.5]" \
        --d_state "[32, 48, 32]" \
        --ssd_expand "[0.5, 4, 4]" \
        --depth "[[1, 0], [0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0]]"
        
sleep 3
# base
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_kitti/run2_searched_B/  \
        --num_epochs ${num_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --device ${device} \
        --mlp_ratio "[3.0, 3.0, 3.5, 3.0]" \
        --d_state "[48, 16, 48]" \
        --ssd_expand "[2, 4, 4]" \
        --depth "[[1, 0], [0, 1, 1, 0], [1, 1, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1]]"

sleep 3
# small
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_kitti/run2_searched_S/  \
        --num_epochs ${num_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --device ${device} \
        --mlp_ratio "[3.0, 3.0, 3.0, 3.0]" \
        --d_state "[16, 16, 16]" \
        --ssd_expand "[0.5, 4, 4]" \
        --depth "[[1, 0], [0, 1, 0, 0], [1, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1]]"

sleep 3
# tiny
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_kitti/run2_searched_T/  \
        --num_epochs ${num_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --device ${device} \
        --mlp_ratio "[3.0, 3.0, 3.0, 0.5]" \
        --d_state "[16, 16, 16]" \
        --ssd_expand "[0.5, 4, 2]" \
        --depth "[[1, 0], [0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0]]"

echo "retrain is fine"
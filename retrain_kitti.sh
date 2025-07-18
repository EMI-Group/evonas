#!/bin/bash

base_configs="configs/retrain_kitti.txt"
device="0,1,2,3"

# small
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_kitti/run0_searched_S/  \
        --device ${device} \
        --mlp_ratio "[4.0, 3.5, 3.0, 4.0]" \
        --d_state "[48, 48, 64]" \
        --ssd_expand "[3, 4, 4]" \
        --depth "[[1, 0], [0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1]]"

sleep 3
# base
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_kitti/run0_searched_B/  \
        --device ${device} \
        --mlp_ratio "[3.0, 3.5, 3.5, 4.0]" \
        --d_state "[48, 48, 64]" \
        --ssd_expand "[0.5, 4, 3]" \
        --depth "[[1, 0], [0, 1, 0, 0], [1, 1, 0, 1, 0, 1, 0, 1], [0, 0, 0, 1]]"

sleep 3
# tiny
python MambaDepthNAS/retrain.py "@${base_configs}" \
        --log_directory ./runs/retrain_kitti/run0_searched_T/  \
        --device ${device} \
        --mlp_ratio "[0.5, 1.0, 1.0, 1.0]" \
        --d_state "[64, 48, 16]" \
        --ssd_expand "[0.5, 1, 0.5]" \
        --depth "[[0, 1], [0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0]]"
        
echo "retrain is fine"
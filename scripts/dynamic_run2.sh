#!/bin/bash


python MambaDepthNAS/retrain.py configs/exp06_OFA_retrain/kitti_random/02_retrain_kitti_small.txt

python MambaDepthNAS/retrain.py configs/exp06_OFA_retrain/kitti_remap/02_retrain_kitti_small.txt

echo "this is fine"
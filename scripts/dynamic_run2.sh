#!/bin/bash

python MambaDepthNAS/retrain.py configs/exp04_IN_retrain/28_retrain_nyu_wo_ppm.txt &

python MambaDepthNAS/retrain.py configs/exp04_IN_retrain/29_retrain_nyu_w_ppm.txt &

wait

python MambaDepthNAS/retrain.py configs/exp04_IN_retrain/24_retrain_kitti_base.txt

python MambaDepthNAS/retrain.py configs/exp04_IN_retrain/25_retrain_kitti_crfs.txt

python MambaDepthNAS/retrain.py configs/exp04_IN_retrain/26_retrain_kitti_idisc.txt

python MambaDepthNAS/retrain.py configs/exp04_IN_retrain/27_retrain_kitti_vmamba.txt

wait

python MambaDepthNAS/retrain.py configs/exp04_IN_retrain/30_retrain_kitti_wo_ppm.txt & 

python MambaDepthNAS/retrain.py configs/exp04_IN_retrain/31_retrain_kitti_w_ppm.txt & 

wait

echo "this is fine"
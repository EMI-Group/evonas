#!/bin/bash

python MambaDepthNAS/retrain.py configs/exp03_retrain/02_retrain_nyu_e20.txt

sleep 3
python MambaDepthNAS/retrain.py configs/exp03_retrain/03_retrain_nyu_e10.txt

echo "this is fine"
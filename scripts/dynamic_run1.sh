#!/bin/bash



python MambaDepthNAS/retrain.py configs/exp06_OFA_retrain/nyu/01_retrain_nyu_base.txt

python MambaDepthNAS/retrain.py configs/exp06_OFA_retrain/nyu/02_retrain_nyu_small.txt

python MambaDepthNAS/retrain.py configs/exp06_OFA_retrain/nyu/03_retrain_nyu_tiny.txt

echo "this is fine"
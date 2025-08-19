#!/bin/bash


python MambaDepthNAS/retrain.py configs/exp04_IN_retrain/04_retrain_nyu_lr2e4.txt

python MambaDepthNAS/retrain.py configs/exp04_IN_retrain/05_retrain_nyu_lr4e4.txt

echo "this is fine"
#!/bin/bash

python MambaDepthNAS/retrain.py configs/02_retrain/nyu_random_base.txt

python MambaDepthNAS/retrain.py configs/02_retrain/nyu_random_small.txt

python MambaDepthNAS/retrain.py configs/02_retrain/nyu_random_tiny.txt


echo "this is fine"
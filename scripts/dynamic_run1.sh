#!/bin/bash

python MambaDepthNAS/retrain.py ./configs/02_retrain/nyu_archpool_small.txt

python MambaDepthNAS/retrain.py ./configs/02_retrain/nyu_archpool_tiny.txt

echo "this is fine"
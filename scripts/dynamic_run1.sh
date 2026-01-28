#!/bin/bash

python SegmentNAS/retrain.py SegmentNAS/configs/02_retrain/retrain_supernet_small.txt

python SegmentNAS/retrain.py SegmentNAS/configs/02_retrain/retrain_supernet_base_ft.txt

echo "this is fine"
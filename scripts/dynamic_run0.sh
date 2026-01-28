#!/bin/bash

python SegmentNAS/retrain.py SegmentNAS/configs/02_retrain/retrain_supernet_base.txt

python SegmentNAS/retrain.py SegmentNAS/configs/02_retrain/retrain_supernet_tiny.txt


echo "this is fine"
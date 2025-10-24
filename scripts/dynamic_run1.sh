#!/bin/bash

python DetectionNAS/train.py DetectionNAS/configs/coco_nas.txt 

python DetectionNAS/train.py DetectionNAS/configs/coco_nas_kd.txt 

echo "this is fine"
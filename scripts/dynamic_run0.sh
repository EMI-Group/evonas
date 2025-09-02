#!/bin/bash


python MambaDepthNAS/search.py configs/search/search_nyu.txt

python MambaDepthNAS/search.py configs/search/search_kitti.txt

python MambaDepthNAS/search.py configs/search/search_nyu_remap.txt

python MambaDepthNAS/search.py configs/search/search_nyu_random.txt

python MambaDepthNAS/search.py configs/search/search_kitti_remap.txt

python MambaDepthNAS/search.py configs/search/search_kitti_random.txt



echo "this is fine"
#!/bin/bash

python MambaDepthNAS/search.py configs/search/search_nyu_p100e50.txt

sleep 3
python MambaDepthNAS/search.py configs/search/search_nyu_p50e100.txt

echo "this is fine"
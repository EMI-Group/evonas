#!/bin/bash

python MambaDepthNAS/search.py configs/01_search/search_nyu_archpool.txt

python MambaDepthNAS/search.py configs/01_search/search_nyu_random.txt 

echo "this is fine"
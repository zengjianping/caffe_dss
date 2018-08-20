#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/export/zengjp/anaconda3/lib

python3 run_tutorial.py --max_side=600\
    --image_dir='images'


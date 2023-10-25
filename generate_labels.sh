#!/bin/bash

python3 dataset_generator.py datasets/2D \
   --only-labels \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 127 \
   --scale-step 1 \
   --window-size 168 \
   --step 6 \

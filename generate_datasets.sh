#!/bin/bash

python3 dataset_generator.py "2D_baseline" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 127 \
   --scale-step 1 \
   --window-size 168 \
   --step 6 && \
python3 dataset_generator.py "2D_window_size_small" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 127 \
   --scale-step 1 \
   --window-size 72\
   --step 6 && \
python3 dataset_generator.py "2D_scale_stop_small" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 31 \
   --scale-step 1 \
   --window-size 168 \
   --step 6 && \
python3 dataset_generator.py "2D_step_small" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 127 \
   --scale-step 1 \
   --window-size 168 \
   --step 3 && \
python3 dataset_generator.py "2D_step_large" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 127 \
   --scale-step 1 \
   --window-size 168 \
   --step 12 && \
python3 dataset_generator.py "2D_scale_step_large" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 127 \
   --scale-step 3 \
   --window-size 168 \
   --step 6 

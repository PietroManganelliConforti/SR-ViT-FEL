#!/bin/bash

python3 dataset_generator.py "2D_datasets/2D_baseline" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 32 \
   --scale-step 1 \
   --window-size 168 \
   --step 6 \
   --only-labels && \
python3 dataset_generator.py "2D_datasets/2D_window_size_small" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 32 \
   --scale-step 1 \
   --window-size 72\
   --step 6 \
   --only-labels && \
python3 dataset_generator.py "2D_datasets/2D_scale_stop_small" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 16 \
   --scale-step 1 \
   --window-size 168 \
   --step 6 \
   --only-labels && \
python3 dataset_generator.py "2D_datasets/2D_step_small" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 32 \
   --scale-step 1 \
   --window-size 168 \
   --step 3 \
   --only-labels && \
python3 dataset_generator.py "2D_datasets/2D_step_large" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 32 \
   --scale-step 1 \
   --window-size 168 \
   --step 12 \
   --only-labels && \
python3 dataset_generator.py "2D_datasets/2D_scale_step_large" \
   --scale-type arange \
   --scale-start 1 \
   --scale-stop 32 \
   --scale-step 3 \
   --window-size 168 \
   --step 6 \
   --only-labels

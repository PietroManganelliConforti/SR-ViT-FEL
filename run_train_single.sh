#!/bin/bash
if [ -z "$1" ]
then
    echo "Missing output var"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Missing transform"
    exit 1
fi

if [ -z "$23" ]
then
    echo "Missing task"
    exit 1
fi


python main.py 2D --dataset_path 2D_datasets/2D_baseline --output_var "${1}" --transform "${2}" --mode "${3}" --gpu 0; \
python main.py 2D --dataset_path 2D_datasets/2D_scale_step_large --output_var "${1}" --transform "${2}" --mode "${3}" --gpu 0; \
python main.py 2D --dataset_path 2D_datasets/2D_scale_stop_small --output_var "${1}" --transform "${2}" --mode "${3}" --gpu 0; \
python main.py 2D --dataset_path 2D_datasets/2D_step_large --output_var "${1}" --transform "${2}" --mode "${3}" --gpu 0; \
python main.py 2D --dataset_path 2D_datasets/2D_step_small --output_var "${1}" --transform "${2}" --mode "${3}" --gpu 0; \
python main.py 2D --dataset_path 2D_datasets/2D_window_size_small --output_var "${1}" --transform "${2}" --mode "${3}" --gpu 0; \

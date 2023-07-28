#!/bin/bash

# ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
# ['morlet', 'morlet2', 'ricker']


# Experiments with different wavelets (on GT*)

# GT* ricker
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "ricker" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "C6H6(GT)" "NOx(GT)" "NO2(GT)"

# GT* morlet2
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet2" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "C6H6(GT)" "NOx(GT)" "NO2(GT)"

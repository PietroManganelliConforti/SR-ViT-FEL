#!/bin/bash

for i in 'CO(GT)' 'PT08.S1(CO)' 'C6H6(GT)' 'PT08.S2(NMHC)' 'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH'

do
    python main.py 2D --dataset_path 2D_datasets/2D_baseline --output_var "CO(GT)" --transform "morlet2" --mode "forecasting_simple" --gpu 0 --bs "8" --variables-to-use $i;\
done
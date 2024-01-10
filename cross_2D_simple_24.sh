#!/bin/bash


command="python3 main.py 2D --dataset_path datasets/2D --output_var 'CO(GT)' --transform 'ricker'\
    --mode 'forecasting_lstm' --gpu 0 --bs 8 --cross_val 5 \
    --variables-to-use 'CO(GT)' 'PT08.S1(CO)' 'C6H6(GT)' 'PT08.S2(NMHC)' \
    'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH'"

if [ "$1" = "--do_test" ]
then
    eval $command$1
else
    eval $command
fi

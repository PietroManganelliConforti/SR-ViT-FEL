#!/bin/bash

# ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
# ['morlet', 'morlet2', 'ricker']

# Experiments with different input parameter combinations (always using morlet wavelet)


# Without CO

# GT* morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "C6H6(GT)" "NOx(GT)" "NO2(GT)"

# PT* morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "PT08.S2(NMHC)" "PT08.S3(NOx)" "PT08.S4(NO2)" "PT08.S5(O3)"

# GT*, HT morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "C6H6(GT)" "NOx(GT)" "NO2(GT)" "T" "RH" "AH"

# GT*, HT, PT* morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "C6H6(GT)" "NOx(GT)" "NO2(GT)" "T" "RH" "AH" "PT08.S2(NMHC)" "PT08.S3(NOx)" "PT08.S4(NO2)" "PT08.S5(O3)"

# PT*, HT morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "PT08.S2(NMHC)" "PT08.S3(NOx)" "PT08.S4(NO2)" "PT08.S5(O3)" "T" "RH" "AH"

# HT morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "T" "RH" "AH"


# With CO

# PT*,CO(GT) morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "CO(GT)" "PT08.S2(NMHC)" "PT08.S3(NOx)" "PT08.S4(NO2)" "PT08.S5(O3)"

# CO(GT) morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "CO(GT)"

# PT.CO morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "PT08.S1(CO)"

# GT*,PT.CO morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "PT08.S1(CO)" "C6H6(GT)" "NOx(GT)" "NO2(GT)"

# PT morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "PT08.S1(CO)" "PT08.S2(NMHC)" "PT08.S3(NOx)" "PT08.S4(NO2)" "PT08.S5(O3)"

# PT, HT morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "PT08.S1(CO)" "PT08.S2(NMHC)" "PT08.S3(NOx)" "PT08.S4(NO2)" "PT08.S5(O3)"  "T" "RH" "AH"

# GT morlet
python main.py 2D --dataset_path datasets/2D --output_var "CO(GT)" --transform "morlet" --mode "forecasting_simple" --gpu 0 --bs 8 \
    --variables-to-use "CO(GT)" "C6H6(GT)" "NOx(GT)" "NO2(GT)"

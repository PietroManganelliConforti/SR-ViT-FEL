# GT*
python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'C6H6(GT)' 'NOx(GT)' 'NO2(GT)' --transform 'LSTMLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'C6H6(GT)' 'NOx(GT)' 'NO2(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'C6H6(GT)' 'NOx(GT)' 'NO2(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8


# PT*
#python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'PT08.S2(NMHC)' 'PT08.S3(NOx)' 'PT08.S4(NO2)' 'PT08.S5(O3)' --transform 'LSTMLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
#python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'PT08.S2(NMHC)' 'PT08.S3(NOx)' 'PT08.S4(NO2)' 'PT08.S5(O3)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
#python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'PT08.S2(NMHC)' 'PT08.S3(NOx)' 'PT08.S4(NO2)' 'PT08.S5(O3)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8


# GT*, HT
#python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'C6H6(GT)' 'NOx(GT)' 'NO2(GT)' 'T' 'RH' 'AH' --transform 'LSTMLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
#python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'C6H6(GT)' 'NOx(GT)' 'NO2(GT)' 'T' 'RH' 'AH' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
#python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'C6H6(GT)' 'NOx(GT)' 'NO2(GT)' 'T' 'RH' 'AH' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8


# PT*, CO(GT)
#python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'CO(GT)' 'PT08.S2(NMHC)' 'PT08.S3(NOx)' 'PT08.S4(NO2)' 'PT08.S5(O3)' --transform 'LSTMLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
#python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'CO(GT)' 'PT08.S2(NMHC)' 'PT08.S3(NOx)' 'PT08.S4(NO2)' 'PT08.S5(O3)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
#python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'CO(GT)' 'PT08.S2(NMHC)' 'PT08.S3(NOx)' 'PT08.S4(NO2)' 'PT08.S5(O3)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8


#echo "*************** Running Stacked1DLinear, forecasting_simple ***************"
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 1
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 2
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 4
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 16

#echo "*************** Running Stacked1DLinear, forecasting_advanced ***************"
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 1
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 2
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 4
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 8
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 16

#echo "*************** Running Stacked2DLinear, forecasting_simple ***************"
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 1
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 2
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 4
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 16


#echo "*************** Running Stacked2DLinear, forecasting_advanced ***************"
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 1
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 2
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 4
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 8
#python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 16

python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'C6H6(GT)' 'PT08.S2(NMHC)' 'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH' --transform 'LSTMLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'C6H6(GT)' 'PT08.S2(NMHC)' 'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
python3 main.py 1D --dataset_path '2D_datasets/2D_baseline' --output_var 'CO(GT)' --variables-to-use 'C6H6(GT)' 'PT08.S2(NMHC)' 'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8









echo "*************** Running Stacked1DLinear, forecasting_simple ***************"
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 1
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 2
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 4
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 16

echo "*************** Running Stacked1DLinear, forecasting_advanced ***************"
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 1
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 2
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 4
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 8
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked1DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 16

echo "*************** Running Stacked2DLinear, forecasting_simple ***************"
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 1
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 2
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 4
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 8
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_simple' --bs 16


echo "*************** Running Stacked2DLinear, forecasting_advanced ***************"
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 1
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 2
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 4
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 8
python3 main.py 1D --dataset_path '2D_datasets/2D_scale_stop_small' --output_var 'CO(GT)' --transform 'Stacked2DLinear' --gpu 'False' --mode 'forecasting_advanced' --bs 16







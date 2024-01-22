# Deep Learning and Wavelet Transform for Air Quality Forecasting 

## Details

- Dataset: https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set


## Installation
`pip install -r requirements.txt`


## 2D dataset generation

`./generate_datasets.sh`


## Training and evaluation

- Baseline approach: `./run_train_1D.sh`

- Proposed method:

    - different input parameters combinations (table 1): `./run_train_2D_combos.sh`

    - different mother wavelets (table 2): `./run_train_2D_wavelets.sh`


## docker run with image

`docker run -it --gpus all -v $(pwd):/app --ipc=host --user 1002:1002 piemmec/stackedresnet2image`

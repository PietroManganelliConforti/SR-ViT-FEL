#### CONFIGURAZIONI VIT PRETRAINED E FREZED E AUG ####

python3 main.py 2D_ViT_im --dataset_path datasets/2D --output_var 'CO(GT)' --transform 'ricker'\
    --mode 'forecasting_lstm' --gpu 0 --bs 8\
    --variables-to-use 'CO(GT)' 'PT08.S1(CO)' 'C6H6(GT)' 'PT08.S2(NMHC)' \
    'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH' \
    --pretrained --freeze --cross_val 5 --augmentation 

python3 main.py 2D --dataset_path datasets/2D --output_var 'CO(GT)' --transform 'ricker'\
    --mode 'forecasting_lstm' --gpu 0 --bs 8\
    --variables-to-use 'CO(GT)' 'PT08.S1(CO)' 'C6H6(GT)' 'PT08.S2(NMHC)' \
    'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH' \
    --pretrained --freeze --cross_val 5 --augmentation 


python3 main.py 2D --dataset_path datasets/2D --output_var 'CO(GT)' --transform 'ricker'\
    --mode 'forecasting_lstm' --gpu 0 --bs 8\
    --variables-to-use 'CO(GT)' 'PT08.S1(CO)' 'C6H6(GT)' 'PT08.S2(NMHC)' \
    'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH' \
    --pretrained --freeze --cross_val 5 --augmentation 




#### CONFIGURAZIONI VIT PRETRAINED E FREEZED NO AUG####

python3 main.py 2D --dataset_path datasets/2D --output_var 'CO(GT)' --transform 'ricker'\
    --mode 'ViT_im' --gpu 0 --bs 8\
    --variables-to-use 'CO(GT)' 'PT08.S1(CO)' 'C6H6(GT)' 'PT08.S2(NMHC)' \
    'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH' \
    --pretrained --freeze --cross_val 5

python3 main.py 2D --dataset_path datasets/2D --output_var 'CO(GT)' --transform 'ricker'\
    --mode '2D_ViT_parallel_SR' --gpu 0 --bs 8\
    --variables-to-use 'CO(GT)' 'PT08.S1(CO)' 'C6H6(GT)' 'PT08.S2(NMHC)' \
    'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH' \
    --pretrained --freeze --cross_val 5


python3 main.py 2D --dataset_path datasets/2D --output_var 'CO(GT)' --transform 'ricker'\
    --mode '2D_ViT_SR_feat_in' --gpu 0 --bs 8\
    --variables-to-use 'CO(GT)' 'PT08.S1(CO)' 'C6H6(GT)' 'PT08.S2(NMHC)' \
    'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH' \
    --pretrained --freeze --cross_val 5




#### CONFIGURAZIONI LSTM ####

# python3 main.py 2D --dataset_path datasets/2D --output_var 'CO(GT)' --transform 'ricker'\
#     --mode 'forecasting_lstm' --gpu 0 --bs 8\
#     --variables-to-use 'CO(GT)' 'PT08.S1(CO)' 'C6H6(GT)' 'PT08.S2(NMHC)' \
#     'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH' \
#     --pretrained --freeze --cross_val 5 


# python3 main.py 2D --dataset_path datasets/2D --output_var 'CO(GT)' --transform 'ricker'\
#     --mode 'forecasting_lstm' --gpu 0 --bs 8\
#     --variables-to-use 'CO(GT)' 'PT08.S1(CO)' 'C6H6(GT)' 'PT08.S2(NMHC)' \
#     'NOx(GT)' 'PT08.S3(NOx)' 'NO2(GT)' 'PT08.S4(NO2)' 'PT08.S5(O3)' 'T' 'RH' 'AH' \
#     --pretrained --freeze --cross_val 5 
import os
from matplotlib import transforms as tr
from scipy import signal
from scipy.signal import ricker
import torchvision
import numpy as np
from tqdm import trange
import tqdm
from utils import *
import argparse
import pandas as pd
from natsort import natsorted
import cv2
from StackedResnet import StackedResNet, LSTMSRForecaster, ViTForecaster
from BaselineArchitectures import Stacked2DLinear, Stacked1DLinear, LSTMLinear
from torchsummary import summary
import time
from datasets import *
from augmentation import CWTAugmentation
import json
import random
import torch

class Normalize(object): # not used anymore
    def __init__(self, mean=[0 for i in range(12)], std=[1 for i in range(12)]):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):

        for i in range(len(tensor)):
          
          tensor[i] = (tensor[i] - self.mean[i])/self.std[i]

        return tensor

def collect_data_2D(data_path , transform, device, output_var, train_test_split, 
                    train_val_split, mode, batch_size, variables_to_use, cross_validation_idx, 
                    cross_val_split, augmentation_flag, aug_type): 


    preprocess = None if not augmentation_flag else CWTAugmentation(aug_type)
    
    dataset = Dataset_2D(data_path=data_path, transform=transform, device=device, output_var=output_var, mode=mode, preprocess=preprocess, variable_to_use=variables_to_use)

    len_dataset = len(dataset)
    mase_denom = dataset.mase_denom

    print(f'\nNumero di Training samples: {len_dataset}')

    if cross_validation_idx != -1:

        indices = np.arange(len_dataset)
        
        
        fold_size = len_dataset // cross_val_split
        remaining = len_dataset % cross_val_split
        

        start = fold_size * cross_validation_idx
    
        end = start + fold_size #+ (remaining if cross_val_split == cross_validation_idx+1 else 0)

        test_indices = indices[start:end]

        train_indices = np.concatenate([indices[:start], indices[end:]])
        
            
        fold_train_dataset = torch.utils.data.Subset(dataset, train_indices)
        fold_val_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        train_data_loader = torch.utils.data.DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_data_loader = torch.utils.data.DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_data_loader = torch.utils.data.DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        print(f'There are: {len(fold_train_dataset)} training samples, {len(fold_val_dataset)} validation samples and {len(fold_val_dataset)} test samples')
        print(f'start: {start}, end: {end}, len_dataset: {len_dataset}')
        print(f'Ther are: {len(train_data_loader)*batch_size} training samples, {len(val_data_loader)*batch_size} validation samples and {len(test_data_loader)*batch_size} test samples')

    else :

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*train_test_split), int(len(dataset)*train_test_split)])
        
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset)*train_val_split), int(len(train_dataset)*train_val_split)])
    
        print(f'Ther are: {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples')

        # create dataloaders
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_data_loader, val_data_loader, test_data_loader, mase_denom



def collect_data_2D_lstm(data_path , transform, device, output_var, train_test_split, 
                    train_val_split, mode, batch_size, variables_to_use, cross_validation_idx, 
                    cross_val_split, augmentation_flag, aug_type, include_signals): 


    preprocess = None if not augmentation_flag else CWTAugmentation(aug_type)
    
    dataset = Dataset_2D(
        data_path=data_path,
        transform=transform,
        device=device,
        output_var=output_var,
        mode=mode,
        preprocess=preprocess, # TODO review
        variable_to_use=variables_to_use,
        include_signals=include_signals
    )
    
    len_dataset = len(dataset)
    mase_denom = dataset.mase_denom

    print(f'\nNumero di Training samples: {len_dataset}')

    if cross_validation_idx != -1:

        indices = np.arange(len_dataset)
        
        
        fold_size = len_dataset // cross_val_split
        remaining = len_dataset % cross_val_split
        

        start = fold_size * cross_validation_idx
    
        end = start + fold_size #+ (remaining if cross_val_split == cross_validation_idx+1 else 0)

        test_indices = indices[start:end]

        train_indices = np.concatenate([indices[:start], indices[end:]])
        
            
        fold_train_dataset = torch.utils.data.Subset(dataset, train_indices)
        fold_val_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        train_data_loader = torch.utils.data.DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_data_loader = torch.utils.data.DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_data_loader = torch.utils.data.DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        print(f'There are: {len(fold_train_dataset)} training samples, {len(fold_val_dataset)} validation samples and {len(fold_val_dataset)} test samples')
        print(f'start: {start}, end: {end}, len_dataset: {len_dataset}')
        print(f'Ther are: {len(train_data_loader)*batch_size} training samples, {len(val_data_loader)*batch_size} validation samples and {len(test_data_loader)*batch_size} test samples')

    else :

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*train_test_split), int(len(dataset)*train_test_split)])
        
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset)*train_val_split), int(len(train_dataset)*train_val_split)])
    
        print(f'Ther are: {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples')

        # create dataloaders
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_data_loader, val_data_loader, test_data_loader, mase_denom



def collect_data_1D(data_path, csv_file, device, train_test_split, train_val_split, output_var, mode, batch_size, variables_to_use): 

    dataset = Dataset_1D_raw(data_path, csv_file=csv_file, device=device, output_var=output_var, mode=mode, variables_to_use=variables_to_use)
    mase_denom = dataset.mase_denom

    print(f'\nNumero di Training samples: {len(dataset)}')

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*train_test_split), int(len(dataset)*train_test_split)])
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset)*train_val_split), int(len(train_dataset)*train_val_split)])

    print(f'There are: {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples')

    # create dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_data_loader, val_data_loader, test_data_loader, mase_denom





def train_model(test_name, train_bool, 
                 lr, epochs, train_loader, 
                 val_loader, test_loader,
                 res_path, device, dim, mode, transform, trained_net_path= "",
                 debug = False, variables_to_use=None, out_channel_idx=None,
                 num_output_features=1,pretrained_flag = False, freezed_flag = False,
                 mase_denom = None):

    
    print('TRAIN_MODEL\n\n')

    # Path

    save_path = res_path + '/' + test_name + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
        print("Creating directory for saving models: ", save_path)


    # Setup-train
    torch.cuda.empty_cache()

    best_val_loss, best_val_rel_err = float('inf'), float('inf')
    best_val_mase = float('inf')

    # Build model

    if (dim == '1D'):
        # In 1D args.transform is equal to the architecture name
        if (transform == "Stacked1DLinear"):
            num_input_channels = len(variables_to_use)
            model = Stacked1DLinear(num_input_channels, mode) 
        elif (transform == "Stacked2DLinear"):
            num_input_channels = len(variables_to_use) # Height of the image in input, number of stacked images = 1
            model = Stacked2DLinear(num_input_channels, mode) 
        elif (transform == "LSTMLinear"):
            num_input_channels = len(variables_to_use)  # Number of stacked images in input 
            model = LSTMLinear(num_input_channels, hidden_size=1024, num_layers=2) 
            print (summary(model, (1, num_input_channels, 168)))


    elif (dim == '2D') or (dim.startswith('2D_LSTM')) or ('2D_ViT' in dim):
        model = torchvision.models.resnet18(pretrained=True, progress=True)

        num_input_channels = len(variables_to_use)  # Number of stacked images in input 

        model = StackedResNet(num_input_channels, num_output_features=num_output_features, resnet=model)

        if (dim == '2D_LSTM_SR'):

            if(pretrained_flag): model.load_state_dict(torch.load("StackedResnet_24output/best_valRelerr_model.pth"))

            if (freezed_flag):
                # freeze stacked resnet
                for param in model.parameters():
                    param.requires_grad = False

            model = LSTMSRForecaster(
                stacked_resnet=model,
                outputs=num_output_features,
                channels=num_input_channels,
                num_layers=2,
                hidden_size=512,
                bidirectional=True,
            )

        elif (dim == '2D_ViT_im' or dim == '2D_ViT_parallel_SR' or dim == '2D_ViT_SR_feat_in'):
            
            if(pretrained_flag): model.load_state_dict(torch.load("StackedResnet_24output/best_valRelerr_model.pth"))

            if (freezed_flag):
                # freeze stacked resnet
                for param in model.parameters():
                    param.requires_grad = False

            model = ViTForecaster(model, dim, outputs=24)


    model = model.to(device)


    if trained_net_path != "":
        print("Loading model state dict from ", trained_net_path)
        model.load_state_dict(torch.load(trained_net_path))
        print("Loaded model state dict")

    ret_dict = {
        "losses" : {
            "loss_train" : [],
            "loss_eval" : [],
            "loss_test" : []
        },
        "rel_err" : {
            "rel_err_train":[],
            "rel_err_eval":[],
            "rel_err_test":[]   
        },
        "mase" : {
            "mase_train":[],
            "mase_eval":[],
            "mase_test":[]   
        }
    }
    
    if train_bool:

        print(f"Training for {epochs} epochs...")
        optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad=False)

        for epoch in range(epochs):


            # Training Phase

            model.train()
            # avoid dropout etc. on freezed parts of the model
            if freezed_flag and dim in ['2D', '2D_ViT_im']:
                model.eval()
            elif freezed_flag:
                model.stacked_resnet.eval()

            train_loss = 0
            train_rel_err = 0
            train_mase = 0

            for i, (images, labels) in enumerate(train_loader):
                if dim == '2D_LSTM_SR':
                    images = (images[0].to(device), images[1].to(device))
                else:
                    images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if dim == '2D_LSTM_SR':
                    # signals: (N, 1, C, L)
                    # images: (N, C, H, W)
                    signals, images = images
                    # print(f'shape of signals: {signals.shape}, shape of images: {images.shape}')
                    # signals: (N, L, C)
                    signals = signals.permute(0, 2, 1)
                    # out: (N, 24)
                    out = model(images, signals)
                else:
                    #print (images.shape)
                    out = model(images)

                # NOTE this could break shapes
                if mode != 'forecasting_lstm':
                    out = torch.flatten(out) # era di default, nel caso 2D_24 non serve

                loss = torch.nn.functional.mse_loss(out, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                rel_err = ((out - labels) / labels).abs().mean()
                train_rel_err += rel_err.item()
                
                mase = mase_numerator(out, labels) / mase_denom
                train_mase += mase.item()


            ret_dict["losses"]["loss_train"].append(train_loss/len(train_loader)) 
            ret_dict["rel_err"]["rel_err_train"].append(train_rel_err/len(train_loader))
            ret_dict["mase"]["mase_train"].append(train_mase/len(train_loader))

            # Validation phase
            model.eval()
            
            with torch.no_grad():

                val_loss, val_rel_err, val_mase = evaluate_model(model, val_loader,device, dim, mode, mase_denom) 

                ret_dict["losses"]["loss_eval"].append(val_loss) 
                ret_dict["rel_err"]["rel_err_eval"].append(val_rel_err)
                ret_dict["mase"]["mase_eval"].append(val_mase)
            
            print("[EPOCH "+str(epoch)+"]","Val_loss: ", val_loss, ",  Val_rel_err: ", val_rel_err, ",  Val_mase: ", val_mase)

            save_after_n_epochs = 0 if debug else 15

            if epoch > save_after_n_epochs and val_loss < best_val_loss:

                torch.save(model.state_dict(), save_path + 'best_valLoss_model.pth')
                torch.save(optimizer.state_dict(), save_path + 'state_dict_optimizer.op')
                
                best_val_loss = val_loss
                print('Saving best val_loss model at epoch',epoch," with loss: ",val_loss)

            if epoch > save_after_n_epochs and val_rel_err < best_val_rel_err:

                torch.save(model.state_dict(), save_path + 'best_valRelerr_model.pth')
                torch.save(optimizer.state_dict(), save_path + 'state_dict_optimizer.op')
                
                best_val_rel_err = val_rel_err
                print('Saving best val_rel_err model at epoch: ',epoch," with rel err: ",val_rel_err)

            if epoch > save_after_n_epochs and val_mase < best_val_mase:

                torch.save(model.state_dict(), save_path + 'best_valMase_model.pth')
                torch.save(optimizer.state_dict(), save_path + 'state_dict_optimizer.op')
                
                best_val_mase = val_mase
                print('Saving best val_mase model at epoch: ',epoch," with mase: ",val_mase)

            if epoch % 50 == 0:

                save_plots_and_report(ret_dict, save_path, test_name, False)
    

    print('\n#----------------------#\n#     Test phase       #\n#----------------------#\n\n')
    
    model.load_state_dict(torch.load(save_path + 'best_valLoss_model.pth', map_location=torch.device(device)))

    #model.load_state_dict(torch.load("results_04_12_ViT_feat_puzzle/2D_forecasting_lstm_CO(GT)_ricker_8_['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']/best_valLoss_model.pth"))

    model.eval()
    
    with torch.no_grad():
        test_loss, test_rel_err, test_mase = evaluate_model(model, test_loader,device, dim, mode, mase_denom) 

    ret_dict["losses"]["loss_test"].append(test_loss) #a point
    ret_dict["rel_err"]["rel_err_test"].append(test_rel_err) #a point
    ret_dict["mase"]["mase_test"].append(test_mase) #a point

    print("[TEST] ","test_loss", test_loss, "test_rel_err", test_rel_err, "test_mase", test_mase)

    
    print('\n#----------------------#\n#   Process Completed  #\n#----------------------#\n\n')


    save_plots_and_report(ret_dict, save_path, test_name, False)

    return (ret_dict["losses"]["loss_test"], ret_dict["rel_err"]["rel_err_test"], ret_dict["mase"]["mase_test"]), save_path




def main_1d(args, cross_validation_idx=-1):

    print(args)

    debug = args.do_debug

    device = args.gpu
    device = hardware_check()


    os.environ["CUDA_VISIBLE_DEVICES"] = device
 
    print("GPU IN USO: ", device)

    # Seed #

    seed = 0
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.enabled=False
    
    torch.backends.cudnn.deterministic=True
    
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    ####### ARGS
    
    system_time = time.localtime()

    system_time_string = time.strftime("%m_%d", system_time)

    var_to_use_path = ""
    for var in args.variables_to_use:
        var_to_use_path += var+"_"

    res_path = os.path.join("results_"+system_time_string, var_to_use_path)

    os.makedirs(res_path, exist_ok=True)

    test_name = f'{args.test_name}_{args.dataset_path.split("/")[-1]}_{args.mode}_{args.output_var}_{args.transform}_{args.bs}_{args.variables_to_use}'

    if cross_validation_idx != -1:
        test_name = f'_cross_val_{cross_validation_idx+1}di{args.cross_val}_' + test_name

    test_name = test_name + ("_augmented" if args.augmentation else "")
    test_name = test_name + ("_freezed" if args.freezed else "")
    test_name = test_name + ("_pretrained" if args.pretrained else "" )

    train_bool = not args.do_test

    print("train_bool",train_bool)

    train_val_split = 0.1

    train_test_split = 0.2

    lr = 1e-5

    epoch = 2 if debug else 100

    debug = debug

    trained_net_path = ""

    train_data_loader, val_data_loader, test_data_loader, mase_denom = collect_data_1D(data_path=args.dataset_path, csv_file="AirQuality.csv", device = device, train_test_split=train_test_split, train_val_split=train_val_split, output_var=args.output_var, mode=args.mode, batch_size=args.bs, variables_to_use=args.variables_to_use)

    # Train model

    return train_model(test_name, train_bool, lr, epoch, train_data_loader, val_data_loader, test_data_loader, res_path, device, args.dim, args.mode, args.transform, trained_net_path, debug, args.variables_to_use,pretrained_flag=args.pretrained,freezed_flag=args.freezed, mase_denom = mase_denom)



def main_cross_val(main_f, args):

    ret_arr = []

    paths = [] ##only the last one can be considered

    for i in range(args.cross_val):

        print("Cross validation iteration", i)

        ret_single_run, path = main_f(args, i)

        ret_arr += [ret_single_run]

        paths += [path]
    
        print("Cross validation iteration", i, "results", ret_arr, "\n\n\n")

    mean_loss, mean_acc, mean_mase = np.mean(ret_arr, axis=0)
    
    json_dict = {
        "mean_acc" : str(mean_acc),
        "mean_loss" : str(mean_loss),
        "mean_mase" : str(mean_mase),
        "args" : str(args),
        "paths" :str( paths),
        "ret_arr" : str(ret_arr)
    }

    with open(os.path.join(paths[-1], "cross_val_results.json"), 'w') as f:
        f.write(json.dumps(json_dict, indent=4))

    import csv

    #open return_of_everything and write the results mean_acc, mean_loss, mean_mase, args, paths, ret_arr

    with open("return_of_everything.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow([mean_acc, mean_loss, mean_mase, args.dim, args.augmentation, args.pretrained, args.freezed, ret_arr, paths[-1] ])
        f.close()





def main_2d(args, cross_validation_idx=-1):
    
    print(args)

    debug = args.do_debug

    device = None

    if torch.cuda.is_available():

        device = ("cuda:"+args.gpu)

        num_devices = torch.cuda.device_count()

        current_device = torch.cuda.current_device()

        device_name = torch.cuda.get_device_name(current_device) 

        print("Device name:", device_name, "Device number:", current_device, "Number of devices:", num_devices)
    
    else:
        device = "cpu"

    print("Actual device: ", device)

    os.environ["CUDA_VISIBLE_DEVICES"] = device
 
    print("GPU IN USO: ", device)

    # Seed #

    seed = 0
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.enabled=False
    
    torch.backends.cudnn.deterministic=True
    
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    ####### ARGS

    system_time = time.localtime()

    system_time_string = time.strftime("%d_%m", system_time)

    res_path = "results_"+system_time_string

    os.makedirs(res_path, exist_ok=True, mode=0o777)


    test_name = f'{args.dim}_{args.dataset_path.split("/")[-1]}_{args.mode}_{args.output_var}_{args.transform}_{args.bs}_{args.variables_to_use}'

    if cross_validation_idx != -1:
        test_name = f'_cross_val_{cross_validation_idx+1}di{args.cross_val}_' + test_name
    

    test_name = test_name + ("_augmented" if args.augmentation else "")
    test_name = test_name + ("_freezed" if args.freezed else "")
    test_name = test_name + ("_pretrained" if args.pretrained else "" )
    test_name = test_name + ("_aug_type_"+args.aug_type if args.aug_type else "")

    train_bool = not args.do_test

    print("train_bool",train_bool)

    input_shape = (3, 362, 512)

    train_val_split = 0.1

    train_test_split = 0.2

    lr = 1e-5

    epoch = 2 if debug else 100

    num_output_features = 24 if args.mode == "forecasting_lstm" else 1

    print("num_output_features", num_output_features)

    debug = debug

    data_path = "./data"

    trained_net_path = ""

    data_path = args.dataset_path

    output_var = args.output_var

    transform = args.transform

    batch_size = args.bs

    train_data_loader, val_data_loader, test_data_loader, mase_denom = collect_data_2D(data_path=data_path, transform = transform, device = device, 
                                                                           output_var= output_var, train_test_split=train_test_split,
                                                                            train_val_split=train_val_split, mode=args.mode, batch_size=batch_size,
                                                                            variables_to_use=args.variables_to_use,
                                                                            cross_validation_idx=cross_validation_idx,
                                                                            cross_val_split = args.cross_val, augmentation_flag = args.augmentation,
                                                                            aug_type=args.aug_type)

    # Train model



    return train_model(test_name, train_bool, lr, epoch, train_data_loader, val_data_loader, test_data_loader, res_path, device, args.dim, args.mode, args.transform, trained_net_path, debug, args.variables_to_use, num_output_features=num_output_features,pretrained_flag=args.pretrained,freezed_flag=args.freezed, mase_denom = mase_denom)



def main_2d_lstm(args, cross_validation_idx=-1):
    
    print(args)

    debug = args.do_debug

    device = None

    if torch.cuda.is_available():

        device = ("cuda:"+args.gpu)

        num_devices = torch.cuda.device_count()

        current_device = torch.cuda.current_device()

        device_name = torch.cuda.get_device_name(current_device) 

        print("Device name:", device_name, "Device number:", current_device, "Number of devices:", num_devices)
    
    else:
        device = "cpu"

    print("Actual device: ", device)

    os.environ["CUDA_VISIBLE_DEVICES"] = device
 
    print("GPU IN USO: ", device)

    # Seed #

    seed = 0
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.enabled=False
    
    torch.backends.cudnn.deterministic=True
    
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    ####### ARGS

    system_time = time.localtime()

    system_time_string = time.strftime("%d_%m", system_time)

    res_path = "results_"+system_time_string

    os.makedirs(res_path, exist_ok=True, mode=0o777)


    test_name = f'{args.dim}_{args.dataset_path.split("/")[-1]}_{args.mode}_{args.output_var}_{args.transform}_{args.bs}_{args.variables_to_use}'

    if cross_validation_idx != -1:
        test_name = f'_cross_val_{cross_validation_idx+1}di{args.cross_val}_' + test_name
    

    test_name = test_name + ("_augmented" if args.augmentation else "")
    test_name = test_name + ("_freezed" if args.freezed else "")
    test_name = test_name + ("_pretrained" if args.pretrained else "" )
    test_name = test_name + ("_aug_type_"+args.aug_type if args.aug_type else "")

    train_bool = not args.do_test

    print("train_bool",train_bool)

    input_shape = (3, 362, 512)

    train_val_split = 0.1

    train_test_split = 0.2

    lr = 1e-5

    epoch = 2 if debug else 100

    num_output_features = 24 if args.mode == "forecasting_lstm" else 1

    print("num_output_features", num_output_features)

    debug = debug

    data_path = "./data"

    trained_net_path = ""

    data_path = args.dataset_path

    output_var = args.output_var

    transform = args.transform

    batch_size = args.bs

    train_data_loader, val_data_loader, test_data_loader, mase_denom = collect_data_2D_lstm(
        data_path=data_path, transform = transform, device = device, 
        output_var= output_var, train_test_split=train_test_split,
        train_val_split=train_val_split, mode=args.mode, batch_size=batch_size,
        variables_to_use=args.variables_to_use,
        cross_validation_idx=cross_validation_idx,
        cross_val_split = args.cross_val, augmentation_flag = args.augmentation,
        aug_type=args.aug_type, include_signals=args.dim == '2D_LSTM_SR'
    )

    # Train model
    return train_model(
        test_name, train_bool, lr, epoch, train_data_loader, val_data_loader,
        test_data_loader, res_path, device, args.dim, args.mode, args.transform,
        trained_net_path, debug, args.variables_to_use, num_output_features=num_output_features,
        pretrained_flag=args.pretrained, freezed_flag=args.freezed, mase_denom = mase_denom,
    )




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('dim', choices=["1D", "2D", "2D_LSTM_SR", "2D_ViT_im", "2D_ViT_parallel_SR", "2D_ViT_SR_feat_in"])

    parser.add_argument('--dataset_path', type=str, required=True)

    parser.add_argument('--output_var', type=str, required=True)

    parser.add_argument('--transform', type=str, required=True)

    parser.add_argument('--gpu', type=str, required=True)

    parser.add_argument('--variables-to-use', nargs='+', default=['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'])

    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--do_debug', action='store_true')

    parser.add_argument('--mode', choices=["regression", "forecasting_simple", "forecasting_advanced", "forecasting_lstm"], default="forecasting_advanced")

    parser.add_argument('--bs', type=int, default=4, help='batch size')

    parser.add_argument('--cross_val', type=int, default=1, help='cross validation iterations')

    parser.add_argument('--test_name', type=str, default="")

    parser.add_argument('--pretrained', action='store_true')

    parser.add_argument('--freezed', action='store_true')

    parser.add_argument('--augmentation', action='store_true')

    parser.add_argument('--aug_type', type=str, default="")
                        

    args = parser.parse_args()

    if args.cross_val > 1:

        if args.dim == "1D":
            main_cross_val(main_1d, args)

        elif args.dim == "2D" or '2D_ViT' in args.dim:
            main_cross_val(main_2d, args)

        elif args.dim.startswith("2D_LSTM"):
            main_cross_val(main_2d_lstm, args)

    else:

        if args.dim == "1D":
            main_1d(args)
        elif args.dim == "2D" or '2D_ViT' in args.dim:
            main_2d(args)
        elif args.dim.startswith("2D_LSTM"):
            main_2d_lstm(args)

    


if __name__ == '__main__':
    
    main()


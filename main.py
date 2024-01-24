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
from StackedResnet import StackedResNet, LSTMForecaster, ViTForecaster
from BaselineArchitectures import Stacked2DLinear, Stacked1DLinear, LSTMLinear
from torchsummary import summary
import time
from datasets import *
from augmentation import CWTAugmentation
import json

class Normalize(object): # not used anymore
    def __init__(self, mean=[0 for i in range(12)], std=[1 for i in range(12)]):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):

        for i in range(len(tensor)):
          
          tensor[i] = (tensor[i] - self.mean[i])/self.std[i]

        return tensor

def collect_data_2D(data_path , transform, device, output_var, train_test_split, train_val_split, mode, batch_size, variables_to_use, cross_validation_idx, cross_val_split, augmentation_flag): 


    preprocess = None if not augmentation_flag else CWTAugmentation()
    
    dataset = Dataset_2D(data_path=data_path, transform=transform, device=device, output_var=output_var, mode=mode, preprocess=preprocess, variable_to_use=variables_to_use)
    
    len_dataset = len(dataset)

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
        
        train_data_loader = torch.utils.data.DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_data_loader = torch.utils.data.DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        test_data_loader = torch.utils.data.DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        print(f'There are: {len(fold_train_dataset)} training samples, {len(fold_val_dataset)} validation samples and {len(fold_val_dataset)} test samples')
        print(f'start: {start}, end: {end}, len_dataset: {len_dataset}')
        print(f'Ther are: {len(train_data_loader)*batch_size} training samples, {len(val_data_loader)*batch_size} validation samples and {len(test_data_loader)*batch_size} test samples')

    else :

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*train_test_split), int(len(dataset)*train_test_split)])
        
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset)*train_val_split), int(len(train_dataset)*train_val_split)])
    
        print(f'Ther are: {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples')

        # create dataloaders
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_data_loader, val_data_loader, test_data_loader



def collect_data_2D_lstm(data_path , transform, device, output_var, train_test_split, train_val_split, mode, batch_size, variables_to_use, cross_validation_idx, cross_val_split, csv_file, augmentation_flag):


    preprocess = None if not augmentation_flag else CWTAugmentation()
    
    dataset = Dataset_2D(data_path=data_path, transform=transform, device=device, output_var=output_var, mode=mode, preprocess=preprocess, variable_to_use=variables_to_use)
    #print(list(dataset.windows[0].keys()))
    output_var_idx = list(dataset.windows[0].keys()).index(output_var)

    len_dataset = len(dataset)


    print(f'\nNumero di Training samples: {len(dataset)}')


    train_dataset_2D, test_dataset_2D = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*train_test_split), int(len(dataset)*train_test_split)])
    
    train_dataset_2D, val_dataset_2D = torch.utils.data.random_split(train_dataset_2D, [len(train_dataset_2D) - int(len(train_dataset_2D)*train_val_split), int(len(train_dataset_2D)*train_val_split)])

    print(f'Ther are: {len(train_dataset_2D)} training samples, {len(val_dataset_2D)} validation samples and {len(test_dataset_2D)} test samples')

    dataset_1D = Dataset_1D_raw(data_path, csv_file=csv_file, device=device, output_var=output_var, mode=mode, variables_to_use=variables_to_use)
                                                                
    print(f'\nNumero di Training samples: {len(dataset)}')

    train_dataset_1D, test_dataset_1D = torch.utils.data.random_split(dataset_1D, [len(dataset_1D) - int(len(dataset_1D)*train_test_split), int(len(dataset_1D)*train_test_split)])
    
    train_dataset_1D, val_dataset_1D = torch.utils.data.random_split(train_dataset_1D, [len(train_dataset_1D) - int(len(train_dataset_1D)*train_val_split), int(len(train_dataset_1D)*train_val_split)])
    
    print(f'There are: {len(train_dataset_1D)} training samples, {len(val_dataset_1D)} validation samples and {len(test_dataset_1D)} test samples')

    train_dataset = ZipDatasets(train_dataset_1D, train_dataset_2D)
    val_dataset = ZipDatasets(val_dataset_1D, val_dataset_2D)
    test_dataset = ZipDatasets(test_dataset_1D, test_dataset_2D)

    if cross_validation_idx != -1:

        dataset_zipped = ZipDatasets(dataset_1D,dataset)

        indices = np.arange(len_dataset) #Da mettere sullo zipdataset?
        
        
        fold_size = len_dataset // cross_val_split
        remaining = len_dataset % cross_val_split
        

        start = fold_size * cross_validation_idx
    
        end = start + fold_size #+ (remaining if cross_val_split == cross_validation_idx+1 else 0)

        test_indices = indices[start:end]

        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        fold_train_dataset = torch.utils.data.Subset(dataset_zipped, train_indices)
        fold_val_dataset = torch.utils.data.Subset(dataset_zipped, test_indices)
        
        train_data_loader = torch.utils.data.DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_data_loader = torch.utils.data.DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        test_data_loader = torch.utils.data.DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        print(f'There are: {len(fold_train_dataset)} training samples, {len(fold_val_dataset)} validation samples and {len(fold_val_dataset)} test samples')
        print(f'start: {start}, end: {end}, len_dataset: {len_dataset}')
        print(f'Ther are: {len(train_data_loader)*batch_size} training samples, {len(val_data_loader)*batch_size} validation samples and {len(test_data_loader)*batch_size} test samples')
    
    else:
        
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_data_loader, val_data_loader, test_data_loader, output_var_idx





def collect_data_1D(data_path, csv_file, device, train_test_split, train_val_split, output_var, mode, batch_size, variables_to_use): 

    dataset = Dataset_1D_raw(data_path, csv_file=csv_file, device=device, output_var=output_var, mode=mode, variables_to_use=variables_to_use)
                                                                 
    print(f'\nNumero di Training samples: {len(dataset)}')

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*train_test_split), int(len(dataset)*train_test_split)])
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset)*train_val_split), int(len(train_dataset)*train_val_split)])

    print(f'There are: {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples')

    # create dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_data_loader, val_data_loader, test_data_loader





def train_model(test_name, train_bool, 
                 lr, epochs, train_loader, 
                 val_loader, test_loader,
                 res_path, device, dim, mode, transform, trained_net_path= "",
                 debug = False, variables_to_use=None, out_channel_idx=None,
                 num_output_features=1,pretrained_flag = False, freezed_flag = False):

    
    print('TRAIN_MODEL\n\n')

    # Path

    save_path = res_path + '/' + test_name + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Creating directory for saving models: ", save_path)


    # Setup-train
    torch.cuda.empty_cache()

    best_val_loss, best_val_rel_err = float('inf'), float('inf')

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


    elif (dim == '2D') or (dim == '2D_LSTM') or ('2D_ViT' in dim):
        model = torchvision.models.resnet18(pretrained=True, progress=True)

        num_input_channels = len(variables_to_use)  # Number of stacked images in input 

        model = StackedResNet(num_input_channels, num_output_features=num_output_features, resnet=model)

        if (dim == '2D_LSTM'):
            # freeze stacked resnet
            for param in model.parameters():
                param.requires_grad = False

            model = LSTMForecaster(model, channels=num_input_channels, num_layers=2, hidden_size=512, outputs=1, mode='option1')

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
        }
    }
    
    if train_bool:

        print(f"Training for {epochs} epochs...")
        optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad=False)

        for epoch in range(epochs):


            # Training Phase

            model.train()
            if dim == '2D_LSTM':
                model.stacked_resnet.eval()

            train_loss = 0
            train_rel_err = 0

            for i, (images, labels) in enumerate(train_loader):


                # if dim == '2D_LSTM':
                #     images, signals = images
                #     cwt = signal.cwt(signals, ricker
                #                 , np.arange(1, 127))
                #     if not torch.all(cwt == images):
                #         print("CWT and images are different")
                #         raise Exception()

                if dim == '2D_LSTM':
                    images = (images[0].to(device), images[1].to(device))        
                else:
                    images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if dim == '2D_LSTM':
                    # signals: (N, L, C)
                    # images: (N, H, W, C)
                    signals, images = images
                    # TODO fix shapes
                    signals = signals.squeeze().reshape(signals.shape[0], signals.shape[2], signals.shape[1])
                    #print(signals.shape, images.shape)
                    # out: (N, 24)
                    out = torch.zeros((images.shape[0], 24)).to(device)
                    for j in range(24):
                        # out[:, j]: (N, 1)
                        out[:,j] = model(images, signals)
                        signals = torch.cat([signals[:, 1:, out_channel_idx], out[:,j].unsqueeze(1)], dim=1)
                else:
                    #print (images.shape)
                    out = model(images)

                    
                if not ( (dim=='2D' or '2D_ViT' in dim) and mode=='forecasting_lstm'):
                    out = torch.flatten(out) # era di default, nel caso 2D_24 non serve

                loss = torch.nn.functional.mse_loss(out, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                rel_err = ((out - labels) / labels).abs().mean()
                train_rel_err += rel_err.item()


            ret_dict["losses"]["loss_train"].append(train_loss/len(train_loader)) 
            ret_dict["rel_err"]["rel_err_train"].append(train_rel_err/len(train_loader)) 

            # Validation phase
            model.eval()
            
            with torch.no_grad():

                val_loss, val_rel_err = evaluate_model(model, val_loader,device, dim, mode) 

                ret_dict["losses"]["loss_eval"].append(val_loss) 
                ret_dict["rel_err"]["rel_err_eval"].append(val_rel_err) 
            
            print("[EPOCH "+str(epoch)+"]","Val_loss: ", val_loss, ",  Val_rel_err: ", val_rel_err)

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

            if epoch % 50 == 0:

                save_plots_and_report(ret_dict, save_path, test_name, False)
    

    print('\n#----------------------#\n#     Test phase       #\n#----------------------#\n\n')
    
    model.load_state_dict(torch.load(save_path + 'best_valLoss_model.pth', map_location=torch.device(device)))

    #model.load_state_dict(torch.load("results_04_12_ViT_feat_puzzle/2D_forecasting_lstm_CO(GT)_ricker_8_['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']/best_valLoss_model.pth"))

    model.eval()
    
    with torch.no_grad():
        test_loss, test_rel_err = evaluate_model(model, test_loader,device, dim, mode) 

    ret_dict["losses"]["loss_test"].append(test_loss) #a point
    ret_dict["rel_err"]["rel_err_test"].append(test_rel_err) #a point

    print("[TEST] ","test_loss", test_loss, "test_rel_err", test_rel_err)

    
    print('\n#----------------------#\n#   Process Completed  #\n#----------------------#\n\n')


    save_plots_and_report(ret_dict, save_path, test_name, False)

    return (ret_dict["losses"]["loss_test"], ret_dict["rel_err"]["rel_err_test"]), save_path




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

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

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

    train_data_loader, val_data_loader, test_data_loader = collect_data_1D(data_path=args.dataset_path, csv_file="AirQuality.csv", device = device, train_test_split=train_test_split, train_val_split=train_val_split, output_var=args.output_var, mode=args.mode, batch_size=args.bs, variables_to_use=args.variables_to_use)

    # Train model

    return train_model(test_name, train_bool, lr, epoch, train_data_loader, val_data_loader, test_data_loader, res_path, device, args.dim, args.mode, args.transform, trained_net_path, debug, args.variables_to_use,pretrained_flag=args.pretrained,freezed_flag=args.freezed)



def main_cross_val(main_f, args):

    ret_arr = []

    paths = [] ##only the last one can be considered

    for i in range(args.cross_val):

        print("Cross validation iteration", i)

        ret_single_run, path = main_f(args, i)

        ret_arr += [ret_single_run]

        paths += [path]
    
        print("Cross validation iteration", i, "results", ret_arr, "\n\n\n")

    mean_acc, mean_loss = np.mean(ret_arr, axis=0)
    
    json_dict = {
        "mean_acc" : str(mean_acc),
        "mean_loss" : str(mean_loss),
        "args" : str(args),
        "paths" :str( paths),
        "ret_arr" : str(ret_arr)
    }

    with open(os.path.join(paths[-1], "cross_val_results.json"), 'w') as f:
        f.write(json.dumps(json_dict, indent=4))

    import csv

    #open return_of_everything and write the results mean_acc, mean_loss, args, paths, ret_arr

    with open("return_of_everything.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow([mean_acc, mean_loss, args, paths, ret_arr])
        f.close()





def main_2d(args, cross_validation_idx=-1):
    
    print(args)

    debug = args.do_debug

    device = args.gpu

    device = hardware_check()


    os.environ["CUDA_VISIBLE_DEVICES"] = device
 
    print("GPU IN USO: ", device)

    # Seed #

    seed = 0

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    ####### ARGS

    system_time = time.localtime()

    system_time_string = time.strftime("%d_%m", system_time)

    res_path = "results_"+system_time_string

    os.makedirs(res_path, exist_ok=True)


    test_name = f'{args.test_name}_{args.dataset_path.split("/")[-1]}_{args.mode}_{args.output_var}_{args.transform}_{args.bs}_{args.variables_to_use}'

    if cross_validation_idx != -1:
        test_name = f'_cross_val_{cross_validation_idx+1}di{args.cross_val}_' + test_name
    

    test_name = test_name + ("_augmented" if args.augmentation else "")
    test_name = test_name + ("_freezed" if args.freezed else "")
    test_name = test_name + ("_pretrained" if args.pretrained else "" )

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

    train_data_loader, val_data_loader, test_data_loader = collect_data_2D(data_path=data_path, transform = transform, device = device, 
                                                                           output_var= output_var, train_test_split=train_test_split,
                                                                            train_val_split=train_val_split, mode=args.mode, batch_size=batch_size,
                                                                            variables_to_use=args.variables_to_use,
                                                                            cross_validation_idx=cross_validation_idx,
                                                                            cross_val_split = args.cross_val, augmentation_flag = args.augmentation)

    # Train model



    return train_model(test_name, train_bool, lr, epoch, train_data_loader, val_data_loader, test_data_loader, res_path, device, args.dim, args.mode, args.transform, trained_net_path, debug, args.variables_to_use, num_output_features=num_output_features,pretrained_flag=args.pretrained,freezed_flag=args.freezed)



def main_2d_lstm(args, cross_validation_idx=-1):
    
    print(args)

    debug = args.do_debug

    device = args.gpu

    device = hardware_check()


    os.environ["CUDA_VISIBLE_DEVICES"] = device
 
    print("GPU IN USO: ", device)

    # Seed #

    seed = 0

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    ####### ARGS

    system_time = time.localtime()

    system_time_string = time.strftime("%d_%m", system_time)

    res_path = "results_"+system_time_string

    os.makedirs(res_path, exist_ok=True)

    test_name = f'{args.test_name}_{args.dataset_path.split("/")[-1]}_{args.mode}_{args.output_var}_{args.transform}_{args.bs}_{args.variables_to_use}'

    if cross_validation_idx != -1:
        test_name = f'_cross_val_{cross_validation_idx+1}di{args.cross_val}_' + test_name
    

    test_name = test_name + ("_augmented" if args.augmentation else "")
    test_name = test_name + ("_freezed" if args.freezed else "")
    test_name = test_name + ("_pretrained" if args.pretrained else "" )

    train_bool = not args.do_test

    print("train_bool",train_bool)

    input_shape = (3, 362, 512)

    train_val_split = 0.1

    train_test_split = 0.2

    lr = 1e-5

    epoch = 2 if debug else 100

    debug = debug

    data_path = "./data"

    trained_net_path = ""

    data_path = args.dataset_path

    output_var = args.output_var

    transform = args.transform

    batch_size = args.bs


    train_data_loader, val_data_loader, test_data_loader, output_var_idx = collect_data_2D_lstm(data_path=data_path, transform = transform, device = device, output_var= output_var,
                                                                                                train_test_split=train_test_split, train_val_split=train_val_split, mode=args.mode,
                                                                                                batch_size=batch_size, variables_to_use=args.variables_to_use, 
                                                                                                cross_validation_idx=cross_validation_idx,
                                                                                                cross_val_split = args.cross_val,
                                                                                                csv_file="AirQuality.csv", augmentation_flag = args.augmentation)

    # Train model

    return train_model(test_name, train_bool, lr, epoch, train_data_loader, val_data_loader, test_data_loader, res_path, device, args.dim, args.mode, args.transform, trained_net_path, debug, args.variables_to_use, out_channel_idx=output_var_idx,pretrained_flag=args.pretrained,freezed_flag=args.freezed)




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('dim', choices=["1D", "2D", "2D_LSTM", "2D_ViT_im", "2D_ViT_parallel_SR", "2D_ViT_SR_feat_in"])

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
                        

    args = parser.parse_args()

    if args.cross_val > 1:

        if args.dim == "1D":
            main_cross_val(main_1d, args)

        elif args.dim == "2D" or '2D_ViT' in args.dim:
            main_cross_val(main_2d, args)

        elif args.dim == "2D_LSTM":
            main_cross_val(main_2d_lstm, args)

    else:

        if args.dim == "1D":
            main_1d(args)
        elif args.dim == "2D" or '2D_ViT' in args.dim:
            main_2d(args)
        elif args.dim == "2D_LSTM":
            main_2d_lstm(args)

    


if __name__ == '__main__':
    
    main()


import os
from matplotlib import transforms as tr
import torchvision
import numpy as np
from tqdm import trange
import tqdm
from utils import *
import argparse
import pandas as pd
from natsort import natsorted
import cv2
from StackedResnet import StackedResNet
from BaselineArchitectures import Stacked2DLinear, Stacked1DLinear, LSTMLinear
from torchsummary import summary
import time
from datasets import *

class Normalize(object): # not used anymore
    def __init__(self, mean=[0 for i in range(12)], std=[1 for i in range(12)]):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):

        for i in range(len(tensor)):
          
          tensor[i] = (tensor[i] - self.mean[i])/self.std[i]

        return tensor

def collect_data_2D(data_path , transform, device, output_var, train_test_split, train_val_split, mode, batch_size, variables_to_use): 



    # preprocess = tr.Compose([
    #                             Normalize(0,1)
    #                         ])

    preprocess = None
    
    dataset = Dataset_2D(data_path=data_path, transform=transform, device=device, output_var=output_var, mode=mode, preprocess=preprocess, variable_to_use=variables_to_use)

    print(f'\nNumero di Training samples: {len(dataset)}')

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*train_test_split), int(len(dataset)*train_test_split)])
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset)*train_val_split), int(len(train_dataset)*train_val_split)])
    
    print(f'Ther are: {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples')

    # create dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_data_loader, val_data_loader, test_data_loader




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
                 debug = False, variables_to_use=None):

    
    print('TRAIN_MODEL\n\n')

    # Path

    save_path = res_path + '/' + test_name + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)


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
            num_input_channels = 1  # Number of stacked images in input 
            model = Stacked2DLinear(num_input_channels, mode) 
        elif (transform == "LSTMLinear"):
            num_input_channels = len(variables_to_use)  # Number of stacked images in input 
            model = LSTMLinear(num_input_channels, hidden_size=1024, num_layers=2) 
            print (summary(model, (1, num_input_channels, 168)))


    elif (dim == '2D'):
        model = torchvision.models.resnet18(pretrained=True, progress=True)

        num_input_channels = len(variables_to_use)  # Number of stacked images in input 
        print(num_input_channels)
        model = StackedResNet(num_input_channels, model) # TODO: try with freezed resnet


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

            train_loss = 0
            train_rel_err = 0

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                out = model(images) 
                optimizer.zero_grad()
                out = torch.flatten(out)             
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

                val_loss, val_rel_err = evaluate_model(model, val_loader,device) 

                ret_dict["losses"]["loss_eval"].append(val_loss) 
                ret_dict["rel_err"]["rel_err_eval"].append(val_rel_err) 
            
            print("[EPOCH "+str(epoch)+"]","Val_loss: ", val_loss)

            if epoch > 49 and val_loss < best_val_loss:

                torch.save(model.state_dict(), save_path + 'best_valLoss_model.pth')
                torch.save(optimizer.state_dict(), save_path + 'state_dict_optimizer.op')
                
                best_val_loss = val_loss
                print('Saving best val_loss model at epoch',epoch," with loss: ",val_loss)

            if epoch > 49 and val_rel_err < best_val_rel_err:

                torch.save(model.state_dict(), save_path + 'best_valRelerr_model.pth')
                torch.save(optimizer.state_dict(), save_path + 'state_dict_optimizer.op')
                
                best_val_rel_err = val_rel_err
                print('Saving best val_rel_err model at epoch: ',epoch," with rel err: ",val_rel_err)

            if epoch % 50 == 0:

                save_plots_and_report(ret_dict, save_path, test_name, True)
    

    print('\n#----------------------#\n#     Test phase       #\n#----------------------#\n\n')

    model.eval()
    
    with torch.no_grad():
        test_loss, test_rel_err = evaluate_model(model, test_loader,device) 

    ret_dict["losses"]["loss_test"].append(test_loss) #a point
    ret_dict["rel_err"]["rel_err_test"].append(test_rel_err) #a point

    print("[TEST] ","test_loss", test_loss, "test_rel_err", test_rel_err)

    
    print('\n#----------------------#\n#   Process Completed  #\n#----------------------#\n\n')


    save_plots_and_report(ret_dict, save_path, test_name,True)




def main_1d(args):
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

    test_name = f'{args.dataset_path.split("/")[-1]}_{args.mode}_{args.output_var}_{args.transform}_{args.bs}'

    train_bool = not args.do_test

    print("train_bool",train_bool)

    train_val_split = 0.1

    train_test_split = 0.2

    lr = 1e-5

    epoch = 100

    debug = debug

    trained_net_path = ""

    train_data_loader, val_data_loader, test_data_loader = collect_data_1D(data_path=args.dataset_path, csv_file="AirQuality.csv", device = device, train_test_split=train_test_split, train_val_split=train_val_split, output_var=args.output_var, mode=args.mode, batch_size=args.bs, variables_to_use=args.variables_to_use)

    # Train model

    train_model(test_name, train_bool, lr, epoch, train_data_loader, val_data_loader, test_data_loader, res_path, device, args.dim, args.mode, args.transform, trained_net_path, debug, args.variables_to_use)


def main_2d(args):
    
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

    test_name = f'{args.dataset_path.split("/")[-1]}_{args.mode}_{args.output_var}_{args.transform}_{args.bs}_{args.variables_to_use}'

    train_bool = not args.do_test

    print("train_bool",train_bool)

    input_shape = (3, 362, 512)

    train_val_split = 0.1

    train_test_split = 0.2

    lr = 1e-5

    epoch = 100

    debug = debug

    data_path = "./data"

    trained_net_path = ""

    data_path = args.dataset_path

    output_var = args.output_var

    transform = args.transform

    batch_size = args.bs

    train_data_loader, val_data_loader, test_data_loader = collect_data_2D(data_path=data_path, transform = transform, device = device, output_var= output_var, train_test_split=train_test_split, train_val_split=train_val_split, mode=args.mode, batch_size=batch_size, variables_to_use=args.variables_to_use)

    # Train model

    train_model(test_name, train_bool, lr, epoch, train_data_loader, val_data_loader, test_data_loader, res_path, device, args.dim, args.mode, args.transform, trained_net_path, debug, args.variables_to_use)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('dim', choices=["1D", "2D"])

    parser.add_argument('--dataset_path', type=str, required=True)

    parser.add_argument('--output_var', type=str, required=True)

    parser.add_argument('--transform', type=str, required=True)

    parser.add_argument('--gpu', type=str, required=True)

    parser.add_argument('--variables-to-use', nargs='+', default=['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'])

    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--do_debug', action='store_true')

    parser.add_argument('--mode', choices=["regression", "forecasting_simple", "forecasting_advanced"], default="forecasting_advanced")

    parser.add_argument('--bs', type=int, default=4, help='batch size')

    args = parser.parse_args()

    if args.dim == "1D":
        main_1d(args)
    elif args.dim == "2D":
        main_2d(args)


if __name__ == '__main__':
    
    main()


import os
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
from StackedLinear import Stacked2DLinear, Stacked1DLinear
from torchsummary import summary


def collect_data_2D(data_path , transform, device, output_var, train_test_split, train_val_split, mode): 

    
    dataset = Dataset_2D(data_path=data_path, transform=transform, device=device, output_var=output_var, mode=mode)

    #test_dataset = Dataset_2D(data_path="2D_datasets/2D_scale_step_large", transform="morlet", device=device, output_var="CO(GT)", istrain=True)

    """
    train_dataset = torchvision.datasets.FGVCAircraft(root=data_path, split='trainval', transform=torchvision.transforms.Compose([torchvision.transforms.Resize((input_shape[1], input_shape[2]), antialias=True),
                                                                                                                                    torchvision.transforms.AutoAugment(),
                                                                                                                                    torchvision.transforms.ToTensor(),                                                                                                                                
                                                                                                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), download=True)
    
    test_dataset = torchvision.datasets.FGVCAircraft(root=data_path, split='test', transform=torchvision.transforms.Compose([torchvision.transforms.Resize((input_shape[1], input_shape[2]), antialias=True),
                                                                                                                               torchvision.transforms.ToTensor(),
                                                                                                                               torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), download=True)
    """

    print(f'\nNumero di Training samples: {len(dataset)}')

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*train_test_split), int(len(dataset)*train_test_split)])
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset)*train_val_split), int(len(train_dataset)*train_val_split)])
    
    print(f'Ther are: {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples')

    # create dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

    return train_data_loader, val_data_loader, test_data_loader



'''
Dataset_1D returns input and output as dictionaries of variables and at each variable is assigned a window of size WINDOW_SIZE

For example:
{'input': {'PT08.S1(CO)': [1360.0, ...], 'NMHC(GT)': [150.0, ...], 'C6H6(GT)': [11.9, ...], 'PT08.S2(NMHC)': [1046.0, ...], 'NOx(GT)': [166.0, ...], 'PT08.S3(NOx)': [1056.0, ...], 'NO2(GT)': [113.0, ...], 'PT08.S4(NO2)': [1692.0, ...], 'PT08.S5(O3)': [1268.0, ...], 'T': [13.6, ...], 'RH': [48.9, ...], 'AH': [0.7578, ...], 'Unnamed: 15': nan, 'Unnamed: 16': nan}}
'''

class Dataset_1D(torch.utils.data.Dataset):

    def __init__(self, csv_file, window_size, step, window_discard_ratio=0.2):

        assert step <= window_size

        self.window_discard_ratio = window_discard_ratio

        df = pd.read_csv(csv_file, sep=';')

        df.drop(['Unnamed: 15','Unnamed: 16'], axis = 1, inplace = True) # Unamed sono Nan, e da 9358 in poi sono NaN
        df = df[0:9357]

        variables = {}
        self.column_means = {}
        for column in df.columns:
            if (column != 'Date' and column != 'Time' and column !='NMHC(GT)'):

                variables[column] = [float(str(elem).replace(',','.')) for elem in df[column].tolist()]
                variables[column] = self.create_windows(variables[column], window_size, step)
                values = np.array(variables[column])
                valid_values = values[values > -100]
                #print(valid_values)
                self.column_means[column] = np.mean(valid_values)
                print(f'Column {column} has mean {self.column_means[column]}')
 
        variables, num_samples = self.preprocess_windows (variables)

        self.input = variables # dictionary of lists of values
        #self.classes = classes
        self.num_samples = num_samples
        self.window_size = window_size
        self.step = step

    def create_windows (self, list_of_values, window_size, step):
        windows = []
        for i in range(0, len(list_of_values), step):
            # do not append windows of sizes less than window_size
            if (len(list_of_values[i:i+window_size]) == window_size):
                window = np.array(list_of_values[i:i+window_size])
                windows.append(window)
                

        return windows
    
    # TODO: sliding windows
    def preprocess_windows (self, variables):
        stack_of_windows = []
        columns = list(variables.keys())
        for column in columns:
            stack_of_windows.append(np.array(variables[column]))

        ### Process the stack of windows ###
        self.forecast_simple_labels = { k: list() for k in columns }
        self.forecast_advanced_labels = { k: list() for k in columns }
        stack = np.stack(stack_of_windows, axis=1)
        print ("Stack shape before preprocessing: ", stack.shape)  # (936, 12, WINDOW_SIZE)
        index_to_remove = []
        for i, windows in enumerate(stack):
            # Each window is a list of values for a specific variable of size WINDOW_SIZE
            for j, window in enumerate(windows):
                window_len = len(window)
                if (-200 in window):
                    count = np.count_nonzero(window == -200)
                    if count == len(window):
                        window[:] = self.column_means[columns[j]]
                    else:
                        ### average over all the elements, except -200 
                        window[window == -200]=np.nan
                        mean = np.nanmean(window)
                        # substitute -200 with mean
                        window[np.isnan(window)] = mean
                    if not (count <= int(self.window_discard_ratio * window_len)):
                        # store the index of the window to be removed from the stack
                        if i not in index_to_remove:
                            index_to_remove.append(i)

        print(f'len(stack): {len(stack)}')
        for i, next_windows in enumerate(stack[1:]):
            # Each window is a list of values for a specific variable of size WINDOW_SIZE
            for j, window in enumerate(next_windows):
                next_window = next_windows[j]
                #fore_list.append(list_of_values[i+window_size])
                win_mean = np.mean(next_window)
                if win_mean < 0:
                    raise ValueError(f"Error: win_mean < 0, number of -200: {np.count_nonzero(next_window == -200)}, number of < 0 {np.count_nonzero(next_window < 0)}, column: {columns[j]}, next_window: {next_window}")
                self.forecast_simple_labels[columns[j]].append(win_mean)
                self.forecast_advanced_labels[columns[j]].append((win_mean + self.column_means[columns[j]]) / 2)

        for c in columns:
            self.forecast_simple_labels[c] = np.array(self.forecast_simple_labels[c])
            self.forecast_advanced_labels[c] = np.array(self.forecast_advanced_labels[c])

        # remove the windows from the stack for all the index_to_remove
        stack = np.delete(stack, index_to_remove, axis=0)

        for c in columns:
            self.forecast_simple_labels[c] = np.delete(self.forecast_simple_labels[c], index_to_remove, axis=0)
            self.forecast_advanced_labels[c] = np.delete(self.forecast_advanced_labels[c], index_to_remove, axis=0)

        print ("Stack shape after preprocessing: ", stack.shape) # (710, 12, WINDOW_SIZE)
        print ("forecast_simple_labels shape after preprocessing: ", self.forecast_simple_labels[columns[0]].shape)

        ### Reconstruct dictionary of input and output from the stack ###
        input_columns = variables.keys()
        
        assert len(input_columns) == stack.shape[1]


        # as output we have a dictionary of variables, each of which is a list of windows of size WINDOW_SIZE
        variables_ = {}
        for stack_idx, column in enumerate(input_columns):
            variables_[column] = stack[:, stack_idx]

        return variables_, stack.shape[0]

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):

        input_dict = {}
        for key in self.input.keys():
            input_dict[key] = self.input[key][idx]

        if idx == self.num_samples - 1:
            fore_simple_dict = None
            fore_advanced_dict = None
        else:
            fore_simple_dict = {}
            fore_advanced_dict = {}
            for key in self.input.keys():
                fore_simple_dict[key] = self.forecast_simple_labels[key][idx]
                fore_advanced_dict[key] = self.forecast_advanced_labels[key][idx]

        item = {'input': input_dict, 'fore_simple': fore_simple_dict, 'fore_advanced': fore_advanced_dict}

        return item



class Dataset_1D_raw(torch.utils.data.Dataset):
    def __init__(self, data_path, csv_file, device, window_size=168, step=6, window_discard_ratio=0.2, output_var="CO(GT)", mode="forecasting_simple"):

        assert step <= window_size
        assert mode in {"forecasting_simple", "forecasting_advanced", "regression"}

        self.window_discard_ratio = window_discard_ratio

        df = pd.read_csv(csv_file, sep=';')

        df.drop(['Unnamed: 15','Unnamed: 16'], axis = 1, inplace = True) # Unamed sono Nan, e da 9358 in poi sono NaN
        df = df[0:9357]
        df_len = len(df)
        

        input_variables = {}
        for column in df.columns:
            if (column != 'Date' and column != 'Time' and column !='NMHC(GT)'):
                input_variables[column] = [float(str(elem).replace(',','.')) for elem in df[column].tolist()]
                input_variables[column] = self.create_windows(input_variables[column], window_size, step)
            

        if (mode == "forecasting_simple"):
            label_file = "fore_simple_labels.txt"
            
        elif (mode == "forecasting_advanced"):
            label_file = "fore_advanced_labels.txt"

        # INPUT NEED TO BE FIXED FOR regr_labels 
        elif (mode == "regression"):
            pass
            #label_file = "regr_labels.txt"
            #input_variables.pop(output_var)            

        column = "forelabels"
        f = open(os.path.join(data_path, output_var, label_file), "r")
        outputs = f.readlines()
        f.close()
        output = [float(str(elem)) for elem in outputs]
        

        
        input_stack, num_samples = self.preprocess_windows (input_variables)


        self.input = input_stack # stack of windows
        self.output = output # dictionary of list of values 
        self.num_samples = num_samples
        self.window_size = window_size
        self.step = step
        self.device = device
        self.mode = mode

    def create_windows (self, list_of_values, window_size, step):
        windows = []
        for i in range(0, len(list_of_values), step):
            # do not append windows of sizes less than window_size
            if (len(list_of_values[i:i+window_size]) == window_size):
                windows.append(np.array(list_of_values[i:i+window_size]))
        return windows
    
    # TODO: sliding windows
    def preprocess_windows (self, input_variables):
        stack_of_windows = []
        for column in input_variables.keys():
            stack_of_windows.append(np.array(input_variables[column]))

        ### Process the stack of windows ###
        stack = np.stack(stack_of_windows, axis=1)
        print ("Stack shape before preprocessing: ", stack.shape)  # (936, 12, WINDOW_SIZE)
        index_to_remove = []
        for i, windows in enumerate(stack):
            # Each window is a list of values for a specific variable of size WINDOW_SIZE
            for window in windows:
                window_len = len(window)
                if (-200 in window):
                    count = np.count_nonzero(window == -200) 
                    if (count <= int(self.window_discard_ratio * window_len)):
                            ### average over all the elements, except -200 
                            window[window == -200]=np.nan
                            mean = np.nanmean(window)
                            # substitute -200 with mean
                            window[np.isnan(window)] = mean
                    else:
                        # store the index of the window to be removed from the stack
                        index_to_remove.append(i)
                        break

        # remove the windows from the stack for all the index_to_remove
        stack = np.delete(stack, index_to_remove, axis=0)
        print ("Stack shape after preprocessing: ", stack.shape) # (710, 12, WINDOW_SIZE)

        return stack, stack.shape[0]

    def __len__(self):

        if self.mode == "forecasting_simple" or self.mode == "forecasting_advanced": return self.num_samples - 1

        if self.mode == "regression": return self.num_samples

        raise Exception()
        
    def __getitem__(self, idx):
        input_windows = torch.FloatTensor(np.array(self.input[idx])).unsqueeze(0).to(self.device) # [channel=1, h=12, w=168]
        output = torch.tensor(float(self.output[idx])).to(self.device)
        
        return input_windows, output














def collect_data_1D(data_path, csv_file, device, train_test_split, train_val_split, output_var, mode): 

    dataset = Dataset_1D_raw(data_path, csv_file=csv_file, device=device, output_var=output_var, mode=mode)
                                                                 
    print(f'\nNumero di Training samples: {len(dataset)}')

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*train_test_split), int(len(dataset)*train_test_split)])
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset)*train_val_split), int(len(train_dataset)*train_val_split)])
    
    print(f'There are: {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples')

    # create dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

    return train_data_loader, val_data_loader, test_data_loader


# We need to return a stack of variables that will fed as input into the CNN + output
# TO DO: insert the output, modify input variable names and insert new arguments in dataset2_D
class Dataset_2D(torch.utils.data.Dataset):
    def __init__(self, data_path, transform, device, output_var, mode="regression"):
        
        assert mode in {"forecasting_simple", "forecasting_advanced", "regression"}
        
        windows = {}
        old_variable_dir = None
        for root, _, files in os.walk(data_path):
            for file in natsorted(files):
                file_to_load = os.path.join(root, file)
                transform_dir = root.split("/")[-1]
                
                if (transform_dir == transform):
                    variable_dir = root.split("/")[-2]
                    # initialize idx of windows if we change variable, and initialize an empty dictionary if idx is not present in windows
                    if (old_variable_dir != variable_dir): idx = 0
                    if (idx not in windows): windows[idx] = {}

                    img = cv2.imread(file_to_load)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                    img = img.astype(np.float32)/255
                    
                    windows[idx][variable_dir] = img
                    old_variable_dir = variable_dir
                    idx+=1

        windows = list(windows.values())
        # split train/test sets
        # In forecasting the last element should be at idx "-2" so the output will be at idx = -1

        
        
        """
        if (istrain):

            self.starting_idx = 0
            windows = windows[self.starting_idx:int(len(windows)*(train_test_split)-1)] if (mode == "forecasting") else windows[self.starting_idx:int(len(windows)*(train_test_split))]
        
        else:
            if (mode == "forecasting"):
                self.starting_idx = int(len(windows)*(train_test_split)-1)
                windows = windows[self.starting_idx:-2]  
            else: 
                self.starting_idx = int(len(windows)*(train_test_split))
                windows = windows[self.starting_idx:-1]
        """

        self.windows = windows
        self.num_samples = len(windows)
        self.device = device
        self.mode = mode
        self.data_path = data_path
        self.output_var = output_var

        if (mode == "forecasting_simple"):
            label_file = os.path.join(self.data_path, self.output_var,"fore_simple_labels.txt")


        elif (mode == "forecasting_advanced"):
            label_file = os.path.join(self.data_path, self.output_var,"fore_advanced_labels.txt")

        # INPUT NEED TO BE FIXED FOR regr_labels 
        elif (mode == "regression"):
            #label_file = os.path.join(self.data_path, self.output_var,"regr_labels.txt")
            pass 

        f = open(label_file, "r")
        outputs = f.readlines()
        f.close()

        self.labels = torch.tensor([float(output.strip()) for output in outputs])


        #self.classes = classes


    def __len__(self):

        if self.mode == "forecasting_simple" or self.mode == "forecasting_advanced": return self.num_samples - 1

        if self.mode == "regression": return self.num_samples

        raise Exception()
        

    def __getitem__(self, idx):

        windows = []
        for key in self.windows[idx].keys():
            windows.append(self.windows[idx][key])

        input = torch.tensor(np.array(windows)) #.to(self.device)
        output = self.labels[idx] #.to(self.device)

        # if (self.mode == "forecasting_simple"):
        #     input = torch.tensor(np.array(windows)) #.to(self.device)

        #     # Labe/Output
        #     label_file = os.path.join(self.data_path, self.output_var,"fore_simple_labels.txt")
        #     f = open(label_file, "r")
        #     outputs = f.readlines()
        #     f.close()

        #     output = torch.tensor(float(outputs[idx].strip())) #.to(self.device)

        #     # If the input is at idx, in forecasting we are taking from the .txt file the starting_idx+idx+1
        #     #output = torch.tensor(float(outputs[self.starting_idx+idx].strip().split()[-1])).to(self.device)

        # elif (self.mode == "forecasting_advanced"):
        #     input = torch.tensor(np.array(windows)) #.to(self.device)

        #     # Labe/Output
        #     label_file = os.path.join(self.data_path, self.output_var,"fore_advanced_labels.txt")
        #     f = open(label_file, "r")
        #     outputs = f.readlines()
        #     f.close()

        #     output = torch.tensor(float(outputs[idx].strip())) #.to(self.device)

        #     # If the input is at idx, in forecasting we are taking from the .txt file the starting_idx+idx+1
        #     #output = torch.tensor(float(outputs[self.starting_idx+idx].strip().split()[-1])).to(self.device)


        # elif (self.mode == "regression"):

        #     input = torch.tensor(np.array(windows)) #.to(self.device)

        #     # Labe/Output
        #     label_file = os.path.join(self.data_path, self.output_var,"regr_labels.txt")
        #     f = open(label_file, "r")
        #     outputs = f.readlines()
        #     f.close()

        #     output = torch.tensor(float(outputs[idx].strip())) #.to(self.device)

        #     # If the input is at idx, in forecasting we are taking from the .txt file the starting_idx+idx+1
        #     #output = torch.tensor(float(outputs[self.starting_idx+idx+1].strip().split()[-1])).to(self.device)
        
        #item = {'input': input, 'output': output}
        return input, output




def train_model(test_name, train_bool, 
                 lr, epochs, train_data_loader, 
                 val_data_loader, test_data_loader,
                 env_path, device, dim, mode, transform, trained_net_path= "",
                 debug = False):

    
    print('TRAIN_MODEL\n\n')

    # Path

    save_path = env_path + 'results/' + test_name + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # Hardware
    #device = hardware_check()

    # Setup-train
    torch.cuda.empty_cache()

    best_val_loss, best_val_rel_err = float('inf'), float('inf')

    # Build model

    if (dim == '1D'):
        # In 1D args.transform is equal to the architecture name
        if (transform == "Stacked1DLinear"):
            num_input_channels = 12
            model = Stacked1DLinear(num_input_channels, mode) 
        elif (transform == "Stacked2DLinear"):
            num_input_channels = 1  # Number of stacked images in input 
            model = Stacked2DLinear(num_input_channels, mode) 
    elif (dim == '2D'):
        model = torchvision.models.resnet34(pretrained=False, progress=True)

        num_input_channels = 12  # Number of stacked images in input 
        model = StackedResNet(num_input_channels, model) #da provare con la resnet freezata e più conv iniziali


    model = model.to(device)
    #summary(model, (1, 12, 168))

    if trained_net_path != "":
        print("Loading model state dict from ", trained_net_path)
        model.load_state_dict(torch.load(trained_net_path))
        print("Loaded model state dict")

    # Data loader
    train_loader = DeviceDataLoader(train_data_loader, device)

    val_loader = DeviceDataLoader(val_data_loader, device)

    test_loader = DeviceDataLoader(test_data_loader, device)    

    ret_dict = inizialize_ret_dict()


    if train_bool:

        print(f"Training for {epochs} epochs...")

        for epoch in range(epochs):

            optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad=False)

            # Training Phase

            model.train()

            train_loss = 0
            train_rel_err = 0

            for images, labels in tqdm.tqdm(train_loader):

                #print(images.shape)

                out = model(images) 
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(out, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                _, preds = torch.max(out, dim=1)
                rel_err = ((out - labels) / labels).abs().mean()
                train_rel_err += rel_err.item()


            ret_dict["losses"]["loss_train"].append(train_loss/len(train_loader)) 
            ret_dict["rel_err"]["rel_err_train"].append(train_rel_err/len(train_loader)) 

            # Validation phase
            model.eval()
            
            with torch.no_grad():

                val_loss, val_rel_err = evaluate_model(model, val_loader) 

                print(val_loss)

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

                save_plots_and_report(ret_dict, save_path, test_name)
    

    print('\n#----------------------#\n#     Test pahse       #\n#----------------------#\n\n')

    model.eval()
    
    with torch.no_grad():
        test_loss, test_rel_err = evaluate_model(model, test_loader) 

    ret_dict["losses"]["loss_test"].append(test_loss) #a point
    ret_dict["rel_err"]["rel_err_test"].append(test_rel_err) #a point

    print("[TEST] ","test_loss", test_loss, "test_rel_err", test_rel_err)

    
    print('\n#----------------------#\n#   Process Completed  #\n#----------------------#\n\n')


    save_plots_and_report(ret_dict, save_path, test_name)




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
    
    os.makedirs("results", exist_ok=True)

    test_name = f'{args.dataset_path.split("/")[-1]}_{args.mode}_{args.output_var}_{args.transform}_test'

    train_bool = not args.do_test

    print("train_bool",train_bool)

    train_val_split = 0.1

    train_test_split = 0.2

    lr = 1e-5

    epoch = 100

    debug = debug
    
    env_path = "./" #project/work on docker

    trained_net_path = ""

    train_data_loader, val_data_loader, test_data_loader = collect_data_1D(data_path=args.dataset_path, csv_file="AirQuality.csv", device = device, train_test_split=train_test_split, train_val_split=train_val_split, output_var=args.output_var, mode=args.mode)

    # Train model

    train_model(test_name, train_bool, lr, epoch, train_data_loader, val_data_loader, test_data_loader, env_path, device, args.dim, args.mode, args.transform, trained_net_path, debug)


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

    os.makedirs("results", exist_ok=True)

    test_name = f'{args.dataset_path.split("/")[-1]}_{args.mode}_{args.output_var}_{args.transform}_test'

    train_bool = not args.do_test

    print("train_bool",train_bool)

    input_shape = (3, 362, 512)

    train_val_split = 0.1

    train_test_split = 0.2

    lr = 1e-5

    epoch = 100

    debug = debug
    
    env_path = "./" #project/work on docker

    data_path = "./data"

    trained_net_path = ""

    data_path = args.dataset_path

    output_var = args.output_var

    transform = args.transform

    train_data_loader, val_data_loader, test_data_loader = collect_data_2D(data_path=data_path, transform = transform, device = device, output_var= output_var, train_test_split=train_test_split, train_val_split=train_val_split, mode=args.mode)

    # Train model

    train_model(test_name, train_bool, lr, epoch, train_data_loader, val_data_loader, test_data_loader, env_path, device, args.dim, args.mode, args.transform, trained_net_path, debug)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('dim', choices=["1D", "2D"])

    parser.add_argument('--dataset_path', type=str, required=True)

    parser.add_argument('--output_var', type=str, required=True)

    parser.add_argument('--transform', type=str, required=True)

    parser.add_argument('--gpu', type=str, required=True)

    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--do_debug', action='store_true')

    parser.add_argument('--mode', choices=["regression", "forecasting_simple", "forecasting_advanced"], default="forecasting_advanced")

    args = parser.parse_args()

    if args.dim == "1D":
        main_1d(args)
    elif args.dim == "2D":
        main_2d(args)


if __name__ == '__main__':
    
    main()


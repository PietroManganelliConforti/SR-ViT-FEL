import os
import torchvision
import numpy as np
from utils import *
import argparse
import pandas as pd
from natsort import natsorted
import cv2
from StackedResnet import StackedResNet

def collect_data_2D(data_path , input_shape, train_val_split): 

    train_dataset = None #todo dataloader 2D

    test_dataset = None #todo dataloader 2D

    train_dataset = torchvision.datasets.FGVCAircraft(root=data_path, split='trainval', transform=torchvision.transforms.Compose([torchvision.transforms.Resize((input_shape[1], input_shape[2]), antialias=True),
                                                                                                                                    torchvision.transforms.AutoAugment(),
                                                                                                                                    torchvision.transforms.ToTensor(),                                                                                                                                
                                                                                                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), download=True)
    
    test_dataset = torchvision.datasets.FGVCAircraft(root=data_path, split='test', transform=torchvision.transforms.Compose([torchvision.transforms.Resize((input_shape[1], input_shape[2]), antialias=True),
                                                                                                                               torchvision.transforms.ToTensor(),
                                                                                                                               torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), download=True)
    
    print(f'Numero di classi: {len(train_dataset.classes)}, \nNumero di Training samples: {len(train_dataset)}, \nNumero di Test sample: {len(test_dataset)}')

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset)*train_val_split), int(len(train_dataset)*train_val_split)])
    print(f'Ther are: {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples')

    # create dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

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
        for column in df.columns:
            if (column != 'Date' and column != 'Time' and column !='NMHC(GT)'):
                variables[column] = [float(str(elem).replace(',','.')) for elem in df[column].tolist()]
                variables[column] = self.create_windows(variables[column], window_size, step)
    
        
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
                windows.append(np.array(list_of_values[i:i+window_size]))
        return windows
    
    # TODO: sliding windows
    def preprocess_windows (self, variables):
        stack_of_windows = []
        for column in variables.keys():
            stack_of_windows.append(np.array(variables[column]))

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
        
        item = {'input': input_dict}

        return item




def collect_data_1D(csv_file, output_variables_names, window_size, train_test_split, train_val_split): 

    train_dataset = Dataset_1D(csv_file=csv_file, output_variables_names=output_variables_names, window_size=window_size, istrain=True, train_test_split=train_test_split)


    test_dataset = Dataset_1D(csv_file=csv_file, output_variables_names=output_variables_names, window_size=window_size, istrain=False, train_test_split=train_test_split)
                                                                 

    print(f'Numero di classi: {len(train_dataset.classes)}, \nNumero di Training samples: {len(train_dataset)}, \nNumero di Test sample: {len(test_dataset)}')

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset)*train_val_split), int(len(train_dataset)*train_val_split)])
  
    print(f'Ther are: {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples')

    # create dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return train_data_loader, val_data_loader, test_data_loader


# We need to return a stack of variables that will fed as input into the CNN + output
# TO DO: insert the output, modify input variable names and insert new arguments in dataset2_D
class Dataset_2D(torch.utils.data.Dataset):
    def __init__(self, data_path, transform, device, output_var, istrain=True, train_test_split=0.8, mode="forecasting"):
        
        assert mode == "forecasting" or mode == "regression"
        
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
                    
                    windows[idx][variable_dir] = img
                    old_variable_dir = variable_dir
                    idx+=1

        windows = list(windows.values())
        # split train/test sets
        if (istrain):
            windows = windows[0:int(len(windows)*(train_test_split))]
        else:
            # In forecasting the last element should be at idx "-2" so the output will be at idx = -1
            if (mode == "forecasting"):
                windows = windows[int(len(windows)*(train_test_split)):-2]  
            elif(mode == "regression"):
                windows = windows[int(len(windows)*(train_test_split)):-1]

        self.windows = windows
        self.num_samples = len(windows)
        self.device = device
        self.mode = mode
        self.data_path = data_path
        self.output_var = output_var
        #self.classes = classes


    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):

        windows = []
        for key in self.windows[idx].keys():
            windows.append(self.windows[idx][key])

        if (self.mode == "forecasting"):
            input = torch.Tensor(windows).to(self.device)

            # Labe/Output
            label_file = os.path.join(self.data_path, self.output_var+".txt")
            f = open(label_file, "r")
            outputs = f.readlines()
            f.close()
            # If the input is at idx, in forecasting we are taking idx+1
            output = torch.tensor(float(outputs[idx+1].strip().split()[-1])).to(self.device)


        elif (self.mode == "regression"):
            # TO DO
            pass
        
        item = {'input': input, 'output': output}
        return item




def train_model(test_name, train_bool, 
                 lr, epochs, train_data_loader, 
                 val_data_loader, test_data_loader,
                 env_path, trained_net_path= "",
                 debug = False):

    
    print('TRAIN_MODEL\n\n')

    # Path

    save_path = env_path + test_name + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # Hardware
    device = hardware_check()

    # Setup-train
    torch.cuda.empty_cache()

    best_val_loss, best_val_acc = float('inf'), 0

    # Build model

    model = torchvision.models.resnet34(pretrained=False, progress=True)

    #num_input_channels = 12  # Number of stacked images in input 
            
    #model = StackedResNet(num_input_channels, model) #da provare con la resnet freezata e più conv iniziali


    model = model.to(device)

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
            train_acc = 0

            for images, labels in train_loader:

                out = model(images) 
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(out, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                _, preds = torch.max(out, dim=1)
                acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
                train_acc += acc.item()


            ret_dict["losses"]["loss_train"].append(train_loss/len(train_loader)) 
            ret_dict["acc"]["acc_train"].append(train_acc/len(train_loader)) 

            # Validation phase
            model.eval()
            
            with torch.no_grad():

                val_loss, val_acc = evaluate_model(model, val_loader) 

                print(val_loss)

                ret_dict["losses"]["loss_eval"].append(val_loss) 
                ret_dict["acc"]["acc_eval"].append(val_acc) 
            
            print("[EPOCH "+str(epoch)+"]","Val_loss: ", val_loss, "val_acc: ", val_acc)

            if epoch > 49 and val_loss < best_val_loss:

                torch.save(model.state_dict(), save_path + 'best_valLoss_model.pth')
                torch.save(optimizer.state_dict(), save_path + 'state_dict_optimizer.op')
                
                best_val_loss = val_loss
                print('Saving best val_loss model at epoch',epoch," with loss: ",val_loss)

            if epoch > 49 and val_acc > best_val_acc:

                torch.save(model.state_dict(), save_path + 'best_valAcc_model.pth')
                torch.save(optimizer.state_dict(), save_path + 'state_dict_optimizer.op')
                
                best_val_acc = val_acc
                print('Saving best val_acc model at epoch: ',epoch," with acc: ",val_acc)

            if epoch % 50 == 0:

                save_plots_and_report(ret_dict, save_path, test_name)
    

    print('\n#----------------------#\n#     Test pahse       #\n#----------------------#\n\n')

    model.eval()
    
    with torch.no_grad():
        test_loss, test_acc = evaluate_model(model, test_loader) 

    ret_dict["losses"]["loss_test"].append(test_loss) #a point
    ret_dict["acc"]["acc_test"].append(test_acc) #a point

    print("[TEST] ","test_loss", test_loss, "test_acc", test_acc)

    
    print('\n#----------------------#\n#   Process Completed  #\n#----------------------#\n\n')


    save_plots_and_report(ret_dict, save_path, test_name)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_debug', action='store_true')

    args = parser.parse_args()
    
    debug = args.do_debug

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
 
    print("GPU IN USO: ", args.gpu)

    # Seed #

    seed = 0

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    '''
    ####### ARGS

    test_name = 'Test_name3'

    train_bool = not args.do_test

    print("train_bool",train_bool)

    input_shape = (3, 362, 512)

    train_val_split = 0.1

    lr = 1e-5

    epoch = 100

    debug = debug
    
    env_path = "./" #project/work on docker

    data_path = "./data"

    trained_net_path = ""

    train_data_loader, val_data_loader, test_data_loader = collect_data_2D(data_path=data_path, input_shape=input_shape, train_val_split=train_val_split)

    # Train model
    
    train_model(test_name, train_bool, lr, epoch, train_data_loader, val_data_loader, test_data_loader, env_path, trained_net_path, debug)
    '''
    '''
    # Usage example of Dataset_1D
    csv_file = "AirQuality.csv"
    window_size = 10
    train_test_split = 0.8
    var = Dataset_1D(csv_file=csv_file, window_size=window_size, step = 10)
    print (var.__getitem__(0))
    '''
    
    device = hardware_check()
    var = Dataset_2D(data_path="2D_baseline", transform="morlet", device=device, output_var="CO(GT)")
    print (var.__getitem__(0))
    


if __name__ == '__main__':
    
    main()


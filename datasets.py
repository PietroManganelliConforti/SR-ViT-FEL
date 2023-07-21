import os
import cv2
import torch
import numpy as np
import pandas as pd
from natsort import natsorted


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
                win_mean = np.mean(next_window[0:24]) # mean of the next day
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
    def __init__(self, data_path, csv_file, device, window_size=168, step=6, window_discard_ratio=0.2, output_var="CO(GT)", mode="forecasting_simple", variables_to_use=[]):

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

        # Remove the parameters from the stack that are not in input_variables
        idx_to_remove = []
        for idx, param in enumerate(input_variables.keys()):
            if (param not in variables_to_use):
                idx_to_remove.append(idx)
        input_stack = np.delete(input_stack, idx_to_remove, axis=1)
        print ("Stack shape after removing parameters: ", input_stack.shape)



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



# We need to return a stack of variables that will fed as input into the CNN + output
# TO DO: insert the output, modify input variable names and insert new arguments in dataset2_D
class Dataset_2D(torch.utils.data.Dataset):
    def __init__(self, data_path, transform, device, output_var, mode="regression", preprocess=None, variable_to_use = []):
        
        assert mode in {"forecasting_simple", "forecasting_advanced", "regression"}

        windows = {}
        old_variable_dir = None
        for root, _, files in os.walk(data_path):
            
            transform_dir = root.split("/")[-1]

            if transform_dir not in ["morlet", "morlet2", "ricker"]: continue

            variable_dir = root.split("/")[-2]
            
            if (variable_dir not in variable_to_use): continue
            
            for file in natsorted(files):
                file_to_load = os.path.join(root, file)
                
                #print(f'transform_dir: {transform_dir}')

                if (transform_dir == transform):

                    # initialize idx of windows if we change variable, and initialize an empty dictionary if idx is not present in windows

                    if (old_variable_dir != variable_dir): idx = 0
                    if (idx not in windows): windows[idx] = {}
                    

                    img = cv2.imread(file_to_load)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                    img = img.astype(np.float32)/255
                    
                    windows[idx][variable_dir] = img
                    old_variable_dir = variable_dir
                    idx+=1

        print(f'windows[0].keys(): {windows[list(windows.keys())[0]].keys()}')
        
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
        self.preprocess = preprocess

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

        if self.preprocess is not None:
            input = self.preprocess(input)

        return input, output
    
    
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


#a = Dataset_2D("2D_datasets/2D_scale_stop_small", "morlet", "cpu", "CO(GT)", mode="forecasting_advanced", preprocess=None, variable_to_remove="AH")


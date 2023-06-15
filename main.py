import os
import torchvision
import numpy as np
from utils import *
import argparse
import pandas as pd


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
Dataset_1D returns input and output as dictionaries of variables

For example:
{'input': {'PT08.S1(CO)': 1360.0, 'NMHC(GT)': 150.0, 'C6H6(GT)': 11.9, 'PT08.S2(NMHC)': 1046.0, 'NOx(GT)': 166.0, 'PT08.S3(NOx)': 1056.0, 'NO2(GT)': 113.0, 'PT08.S4(NO2)': 1692.0, 'PT08.S5(O3)': 1268.0, 'T': 13.6, 'RH': 48.9, 'AH': 0.7578, 'Unnamed: 15': nan, 'Unnamed: 16': nan}, 
'output': {'CO(GT)': 2.6}}
'''

class Dataset_1D(torch.utils.data.Dataset):
    def __init__(self, csv_file, output_variables_names, window_size):

        df = pd.read_csv(csv_file, sep=';')

        df.drop(['Unnamed: 15','Unnamed: 16'], axis = 1, inplace = True) # Unamed sono Nan, e da 9358 in poi sono NaN
        df = df[0:9357]
        

        input_variables, output_variables = {}, {}
        classes = []
        for column in df.columns:
            if (column not in output_variables_names):
                if (column != 'Date' and column != 'Time'):
                    input_variables[column] = [float(str(elem).replace(',','.')) for elem in df[column].tolist()]
                    input_variables[column] = self.create_windows(input_variables[column], window_size)
            else:
                output_variables[column] = [float(str(elem).replace(',','.')) for elem in df[column].tolist()]  
                output_variables[column] = self.create_windows(output_variables[column], window_size)

                classes.append(column)
        
        self.input = input_variables # dictionary of lists of values
        self.output = output_variables # dictionary of list of values 
        self.classes = classes
        self.num_samples = len(df) 
        self.window_size = window_size

    def create_windows (self, list_of_values, window_size):
        windows = []
        for i in range(0, len(list_of_values), window_size):
            windows.append(list_of_values[i:i+window_size])
        return windows
        
    def __len__(self):
        return self.num_samples    
        
    def __getitem__(self, idx):

        input_dict = {}
        for key in self.input.keys():
            input_dict[key] = self.input[key][idx]

        output_dict = {}
        for key in self.output.keys():
            output_dict[key] = self.output[key][idx]

        
        item = {'input': input_dict,
                'output': output_dict}

        return item




def collect_data_1D(data_path , input_shape, train_val_split): 

    train_dataset = None #todo dataset x dataloader 1D

    test_dataset = None #todo dataset x dataloader 1D                                                                            


    print(f'Numero di classi: {len(train_dataset.classes)}, \nNumero di Training samples: {len(train_dataset)}, \nNumero di Test sample: {len(test_dataset)}')

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset)*train_val_split), int(len(train_dataset)*train_val_split)])
  
    print(f'Ther are: {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples')

    # create dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return train_data_loader, val_data_loader, test_data_loader




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
    
    output_variables_names = ['CO(GT)']
    window_size = 10
    var = Dataset_1D(csv_file="AirQuality.csv", output_variables_names=output_variables_names, window_size=window_size)
    print (var.__getitem__(0))
    
if __name__ == '__main__':
    
    main()



import os
import torchvision
import numpy as np
from utils import *
import argparse



def collect_data(root_path , input_shape, train_val_split, seed): 
    
    #QUI E' DOVE ANDRA' IL CODICE PER IL NOSTRO DATALOADER

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_dataset = torchvision.datasets.FGVCAircraft(root=root_path, split='trainval', transform=torchvision.transforms.Compose([torchvision.transforms.Resize((input_shape[1], input_shape[2]), antialias=True),
                                                                                                                                    torchvision.transforms.AutoAugment(),
                                                                                                                                    torchvision.transforms.ToTensor(),                                                                                                                                
                                                                                                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), download=True)
    
    test_dataset = torchvision.datasets.FGVCAircraft(root=root_path, split='test', transform=torchvision.transforms.Compose([torchvision.transforms.Resize((input_shape[1], input_shape[2]), antialias=True),
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




def train_model(test_name, train_bool, 
                 lr, epochs, train_data_loader, 
                 val_data_loader, test_data_loader,
                 root_path, save_path, debug = False):

    
    print('TRAIN_MODEL\n\n')
    # Path
    env_path = root_path

    if not os.path.exists(env_path + test_name + '/'):
        os.makedirs(env_path + test_name + '/')

    save_path = env_path + test_name + '/'

    # Hardware
    device = hardware_check()

    # Setup-train
    torch.cuda.empty_cache()

    best_val_loss, best_val_acc = float('inf'), 0

    # Build model

    model = torchvision.models.resnet34(pretrained=False, progress=True)

    model = model.to(device)

    #model_dict = torch.load(env_path + model_folder + 'best_valAcc_model.pth', map_location = torch.device(device))

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


            ret_dict["losses"]["loss_train"].append(train_loss) 
            ret_dict["acc"]["acc_train"].append(train_acc/len(train_loader)) 

            # Validation phase
            model.eval()
            
            with torch.no_grad():

                val_loss, val_acc = evaluate_model(model, val_loader) 

                ret_dict["losses"]["loss_eval"].append(val_loss) 
                ret_dict["acc"]["acc_eval"].append(val_acc) 

            if epoch > 49 and val_loss < best_val_loss:

                torch.save(model.state_dict(), save_path + 'best_valLoss_model.pth')
                best_val_loss = val_loss
                print('Saving best val_loss model at epoch',epoch," with loss: ",val_loss)

            if epoch > 49 and val_acc > best_val_acc:

                torch.save(model.state_dict(), save_path + 'best_valAcc_model.pth')
                best_val_acc = val_acc
                print('Saving best val_acc model at epoch: ',epoch," with acc: ",val_acc)

            if epoch % 50 == 0:

                save_plot_loss_or_acc( ret_dict["losses"], path = save_path + "/loss/" , test_name = "loss" )
                save_plot_loss_or_acc( ret_dict["acc"], path = save_path + "/acc/" , test_name = "acc" )

    

    print('\n#----------------------#\n#     Test pahse       #\n#----------------------#\n\n')

    model.eval()
    
    with torch.no_grad():
        test_loss, test_acc = evaluate_model(model, test_loader) 


    ret_dict["losses"]["loss_test"].append(test_loss) #a point
    ret_dict["acc"]["acc_test"].append(test_acc) #a point

    
    print('\n#----------------------#\n#   Process Completed  #\n#----------------------#\n\n')

    ret_str ="loss_train: " + str(ret_dict["losses"]["loss_train"][-1:])
    ret_str +="\n\nloss_eval: " + str(ret_dict["losses"]["loss_eval"][-1:]) 
    ret_str +="\n\nloss_test: " + str(ret_dict["losses"]["loss_test"][-1:]) 
    ret_str +="\n\nacc_train " + str(ret_dict["acc"]["acc_train"][-1:])
    ret_str +="\n\nacc_eval: " + str(ret_dict["acc"]["acc_eval"][-1:]) 
    ret_str +="\n\nacc_test: " + str(ret_dict["acc"]["acc_test"][-1:]) 

    with open(save_path+'RESULTS_'+ test_name +'.txt', 'w+') as f:
        f.write(test_name 
        + "\n\n"+ ret_str + '\n\nret dict:\n' + str(ret_dict))

    save_plot_loss_or_acc( ret_dict["losses"], path = save_path + "/loss/" , test_name = "loss" )
    save_plot_loss_or_acc( ret_dict["acc"], path = save_path + "/acc/" , test_name = "acc" )



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_debug', action='store_true')

    args = parser.parse_args()
    
    debug = args.do_debug

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
 
    print("GPU IN USO: ", args.gpu)

    test_name = 'Test_name'

    # Seed
    seed = 0

    # Globals
    train_bool = not args.do_test

    print("train_bool",train_bool)

    input_shape = (3, 362, 512)

    train_val_split = 0.1
    lr = 1e-5
    epoch = 10000
    debug = debug
    
    # Collect data
    root_path = "./data"

    train_data_loader, val_data_loader, test_data_loader = collect_data(root_path=root_path, input_shape=input_shape, train_val_split=train_val_split, seed=seed)

    # Train model

    train_model(test_name, train_bool, lr, epoch, train_data_loader, val_data_loader, test_data_loader, root_path, debug)



if __name__ == '__main__':
    main()


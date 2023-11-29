import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import make_interp_spline

def hardware_check():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Actual device: ", device)
    if 'cuda' in device:
        print("Device info: {}".format(str(torch.cuda.get_device_properties(device)).split("(")[1])[:-1])

    return device



def evaluate_model(model, loader,device, dim, mode):

    loss = 0
    rel_err = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        if not ( (dim=='2D' or '2D_ViT' in dim) and mode=='forecasting_lstm'):
            out = torch.flatten(out)
        loss += torch.nn.functional.mse_loss(out, labels)
        rel_err += ((out - labels) / labels).abs().mean()

    return (loss/len(loader)).item(), (rel_err/len(loader)).item()



def save_plot_loss_or_rel_err(info_dict, n_of_elements, path, test_name, smooth=False):
    """
        ret = {
            "losses" : {
                "loss_train" : [5,4,3,2,1],
                "loss_eval" : [6,5,4,3,2],
                "loss_test" : [2.5]
            },
            "acc" : {
                "acc_train":[10,9,8,7,6],
                "acc_eval":[11,10,9,8,7],
                "acc_test":[7.8]   
            }
        }
    """


    for k in list(info_dict.keys()):
        
        if k == "loss_test" or k == "rel_err_test":
            if len(info_dict[k]) > 0:
                plt.plot(n_of_elements - 1, info_dict[k], '-x', label=k)   # to position text in the background
        
        else:
            if smooth and len(info_dict[k]) > 1:
                idx= range(n_of_elements) 
                spl = make_interp_spline(idx, info_dict[k], k=3)
                xnew = np.linspace(min(idx), max(idx), 300)
                smoothing = spl(xnew)

                plt.plot(xnew, smoothing, '-', label=k)
            else:    
                plt.plot(info_dict[k], '-', label=k)

                
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(  str(list(info_dict.keys())[0]).split("_")[0] + " vs. No. of epochs" )

    if not os.path.exists(path):
        print("PATH: ",path)
        os.makedirs(path)

    if smooth: plt.savefig(path+test_name+"_smoothed.png")
    else: plt.savefig(path+test_name+".png")
    plt.clf()
    plt.close()


    

def save_plots_and_report(ret_dict, save_path, test_name, smooth= False):

    ret_str ="loss_train: " + str(ret_dict["losses"]["loss_train"][-1:])
    ret_str +="\n\nloss_eval: " + str(ret_dict["losses"]["loss_eval"][-1:]) 
    ret_str +="\n\nloss_test: " + str(ret_dict["losses"]["loss_test"][-1:]) 
    ret_str +="\n\nrel_err_train " + str(ret_dict["rel_err"]["rel_err_train"][-1:])
    ret_str +="\n\nrel_err_eval: " + str(ret_dict["rel_err"]["rel_err_eval"][-1:]) 
    ret_str +="\n\nrel_err_test: " + str(ret_dict["rel_err"]["rel_err_test"][-1:]) 

    with open(save_path+test_name +'.txt', 'w+') as f:
        f.write(test_name + "\n\n"+ ret_str + '\n\nret dict:\n' + str(ret_dict))

    n_of_elements = len(ret_dict["losses"]["loss_train"]) 

    save_plot_loss_or_rel_err( ret_dict["losses"], n_of_elements ,path = save_path + "loss/" , test_name = "loss" , smooth= smooth)
    save_plot_loss_or_rel_err( ret_dict["rel_err"], n_of_elements ,path = save_path + "rel_err/" , test_name = "rel_err" , smooth= smooth)

    if smooth:
        save_plot_loss_or_rel_err( ret_dict["losses"], n_of_elements ,path = save_path + "loss/" , test_name = "loss" , smooth= False)
        save_plot_loss_or_rel_err( ret_dict["rel_err"], n_of_elements ,path = save_path + "rel_err/" , test_name = "rel_err" , smooth= False)


"""
test_ret = {
            "losses" : {
                "loss_train" : [5,40,3,20,1],
                "loss_eval" : [6,50,4,30,2],
                "loss_test" : [2.5]
            },
            "rel_err" : {
                "rel_err_train":[10,90,8,70,6],
                "rel_err_eval":[11,100,9,80,7],
                "rel_err_test":[7.8]   
            }
        }

save_plots_and_report(test_ret,"","da_canc", True)
"""


# Define a function to extract feature maps
def get_interested_feature_map(model, x, target_layer):
    # Register a hook to get the feature maps at the desired layer
    feature_map = None

    def hook(module, input, output):
        nonlocal feature_map
        feature_map = output

    hook_handle = target_layer.register_forward_hook(hook)

    # Forward pass to compute the feature maps
    model(x)

    # Remove the hook to prevent it from being called again
    hook_handle.remove()

    return feature_map
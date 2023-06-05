import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def hardware_check():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Actual device: ", device)
    if 'cuda' in device:
        print("Device info: {}".format(str(torch.cuda.get_device_properties(device)).split("(")[1])[:-1])

    return device

def to_device(data, device):
    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    

def save_plot_loss_or_acc(info_dict, path, test_name):
    """

    info_dict = {
                "losses" : {
                    "S_loss" : [],
                    "T_Dist_loss" : [],
                    "KD_train" : [],
                    "XAI_train" : [],
                    "TOT_train" : [],
                    "TOT_eval" : []
                },
                "acc" : {
                    "acc_train":[],
                    "acc_test":[]
                }
            }

    """

    for i in info_dict.keys():
        plt.plot(info_dict[i], '-x', label=i)

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Accuracy vs. No. of epochs')

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path+test_name+".png")
    plt.clf()
    plt.close()



def inizialize_ret_dict():
    ret = {
        "losses" : {
            "loss_train" : [],
            "loss_eval" : [],
            "loss_test" : []
        },
        "acc" : {
            "acc_train":[],
            "acc_eval":[],
            "acc_test":[]   
        }
    }
    return ret


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate_model(model, loader):

    loss = 0
    acc = 0

    for batch in loader:
        images, labels = batch
        out = model(images)

        acc += accuracy(out,labels)
        loss += torch.nn.functional.cross_entropy(out, labels)

    return (loss/len(loader)).item(), (acc/len(loader)).item()



def save_plot_loss_or_acc(info_dict, path, test_name):
    """
        ret = {
            "losses" : {
                "loss_train" : [],
                "loss_eval" : [],
                "loss_test" : []
            },
            "acc" : {
                "acc_train":[],
                "acc_eval":[],
                "acc_test":[]   
            }
        }
    """


    for i in info_dict.keys():
        plt.plot(info_dict[i], '-x', label=i)

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Accuracy vs. No. of epochs')

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path+test_name+".png")
    plt.clf()
    plt.close()
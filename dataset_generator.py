import argparse
import json
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange, tqdm
import os

from main import Dataset_1D


def process_intervals_from_dataset(dataset, wavelets, scales, k, window_size, step, folder_name, only_labels_flag = False):

    cwt_results = {}

    for wavelet in wavelets:
        
        cwtmatr_arr = []

        print("Wavelet: ", wavelet.__name__)

        cwt_results[wavelet.__name__] = []

        f_regr = open(folder_name + "/" + k + "/regr_labels.txt", "w")

        f_fore = open(folder_name + "/" + k + "/fore_labels.txt", "w")

        for i, sample in enumerate(dataset):

            interval = sample["input"][k]

            f_fore.write(str(sample["fore_dict"][k])+"\n")

            f_regr.write(str(np.mean(interval))+"\n")

            if not only_labels_flag: cwtmatr_arr += [signal.cwt(interval, wavelet, scales)]


        f_regr.close()

        f_fore.close()

        if only_labels_flag: continue

        print("Saving..")

        if not os.path.exists(folder_name + "/" + k + "/" + wavelet.__name__):

            os.makedirs(folder_name + "/" + k + "/" + wavelet.__name__)


        for i, cwtmatr in enumerate(tqdm(cwtmatr_arr)):

            plt.imshow(abs(cwtmatr), extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
                    vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
            
            fig_name = folder_name + "/" + k + "/" + wavelet.__name__ + "/" + str(i) + "_"
            
            label = str(i*step) + "_" + str(i*step+window_size) #TODO CON LA LABEL CORRETTA

            fig_name += label

            plt.xticks([])
            plt.yticks([])

            plt.savefig(fig_name, bbox_inches='tight',pad_inches=0.0 )

            plt.clf()



# Parse and save arguments to the command line

command_line = "python " + ' '.join(sys.argv)

# usage: python dataset_generator.py 2D_dataset --scale-type arange --scale-start 1 --scale-stop 127 --scale-step 1 --window-size 168 --step 6
parser = argparse.ArgumentParser(description='Generate 2D dataset')
parser.add_argument('output_dir',)
parser.add_argument('--window-size', type=int, default=24*7,)
parser.add_argument('--step', type=int, default=6,)
parser.add_argument('--scale-type', choices=["arange", "lin", "log"],)
parser.add_argument('--scale-start', type=float)
parser.add_argument('--scale-stop', type=float)
parser.add_argument('--scale-num', default=None)
parser.add_argument('--scale-step', default=None)
parser.add_argument('--only-labels', action='store_true')
args = parser.parse_args()

if args.scale_type == 'arange' and (args.scale_num is not None or args.scale_step is None):
    raise ValueError()
elif args.scale_type != 'arange' and (args.scale_num is None or args.scale_step is not None):
    raise ValueError()

if args.scale_start <= 0:
    raise ValueError()

folder_name = args.output_dir #"2D_Dataset"

if not os.path.exists(folder_name):

    os.makedirs(folder_name)

with open(args.output_dir + '/command_line.txt', 'w') as file:
    file.write(command_line)

# with open(args.output_dir + "/params.json", "w") as f:
#     json.dump(vars(args), f)



# Generate the dataset

dataset = Dataset_1D(
    csv_file='AirQuality.csv',
    window_size=args.window_size,
    step=args.step,
)

# for i, d in enumerate(dataset):
#     obj = d["input"]["CO(GT)"]
#     if i == 0:
#         print(f'{obj[0:100]}')
#     print(i, type(obj), obj.shape)
# print(len(dataset))

wavelets = [
    signal.ricker,
    signal.morlet,
    signal.morlet2,
    # signal.cascade,
    # signal.daub,
    # signal.qmf,
]

scales = {
    "arange": np.arange,
    "lin": np.linspace,
    "log": np.logspace,
}[args.scale_type](args.scale_start, args.scale_stop, args.scale_num).astype(int)
# examples:
# scales = np.arange(1, 127) # avg between 127 and 31
# scales = np.logspace(start=1, stop=127, num=32) # doesn't seem to work

only_labels_flag = args.only_labels

for k in list(next(iter(dataset))["input"].keys()):

    print("Saving param: ", k)

    if not os.path.exists(folder_name + "/" + k ):

        os.makedirs(folder_name + "/" + k)

    process_intervals_from_dataset(dataset, wavelets, scales, k, args.window_size, args.step, folder_name=folder_name, only_labels_flag=only_labels_flag)





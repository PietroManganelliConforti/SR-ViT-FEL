import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange, tqdm
import os



def process_intervals(array, interval_length, wavelets, k, step):

    num_intervals = len(array) // interval_length
    cwt_results = {}
    cwtmatr_arr = []

    for wavelet in wavelets:
        
        print("Wavelet: ", wavelet.__name__)

        cwt_results[wavelet.__name__] = []

        for i in trange(0, len(array) - interval_length, step):

            interval = array[i : i + interval_length ]

            cwtmatr_arr += [signal.cwt(interval, wavelet, np.arange(1, 31))]


        print("Saving..")

        if not os.path.exists(folder_name + "/" + k + "/" + wavelet.__name__):

            os.makedirs(folder_name + "/" + k + "/" + wavelet.__name__)


        for i, cwtmatr in enumerate(tqdm(cwtmatr_arr)):


            plt.imshow(abs(cwtmatr), extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
                    vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
            
            fig_name = folder_name + "/" + k + "/" + wavelet.__name__ + "/" + str(i) + "_"
            
            label = "target" #TODO CON LA LABEL CORRETTA

            fig_name += label

            plt.xticks([])
            plt.yticks([])

            plt.savefig(fig_name, bbox_inches='tight',pad_inches=0.0 )

            plt.clf()

    return cwt_results, num_intervals






# Caricamento del file CSV in un DataFrame di pandas
df = pd.read_csv('AirQuality.csv',sep=";")

df.drop(['Date','Time','Unnamed: 15','Unnamed: 16'], axis = 1, inplace = True) # Unamed sono Nan, e da 9358 in poi sono NaN

df = df[0:9357]

# Creazione di un dizionario in cui ogni chiave è il nome di una colonna e il valore è un array con i dati di quella colonna
arrays = {column: df[column].values for column in df.columns}


wavelets = [signal.ricker, signal.morlet]   

"""
   cascade      -- Compute scaling function and wavelet from coefficients.
   daub         -- Return low-pass.
   morlet       -- Complex Morlet wavelet.
   qmf          -- Return quadrature mirror filter from low-pass.
   ricker       -- Return ricker wavelet.
   morlet2      -- Return Morlet wavelet, compatible with cwt.
   cwt          -- Perform continuous wavelet transform.
"""



folder_name = "2D_Dataset"

if not os.path.exists(folder_name):

    os.makedirs(folder_name)




step = 1000

interval_length = 1000

for k in list(arrays.keys()):

    print("Saving param: ", k)

    if not os.path.exists(folder_name + "/" + k ):

        os.makedirs(folder_name + "/" + k)

    results, num_intervals = process_intervals(arrays[k], interval_length, wavelets, k, step=step)





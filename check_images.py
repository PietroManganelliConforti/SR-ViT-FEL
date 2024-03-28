import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
from tqdm import tqdm
from datasets import Dataset_2D

from main import collect_data_2D_lstm, hardware_check

# Load the images
image1 = cv2.imread(sys.argv[1])
image2 = cv2.imread(sys.argv[2])

# Calculate the pixel-wise difference
print(image1.max())
diff = cv2.absdiff(image1, image2)
print(abs(diff).max())

print(f'All pixels are equal: {np.allclose(image1, image2)}')

# Plot the difference image
plt.imshow(diff)
plt.axis('off')
plt.show()


# def sig2cwtimg(window, wavelet, scales, filename):
#     cwtmatr = scipy.signal.cwt(window, wavelet, scales)
#     plt.imshow(abs(cwtmatr), extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
#             vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig(filename, bbox_inches='tight',pad_inches=0.0 )
#     plt.clf()

# def load_img(filename):
#     img = cv2.imread(filename)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#     img = img.astype(np.float32)/255
#     return img


# # def main2():

# #     data_1d = data_path, csv_file, device, window_size=168, step=6, window_discard_ratio=0.2, output_var="CO(GT)", mode="forecasting_simple", variables_to_use=[])


# def main2():

#     device = hardware_check()

#     dataset = Dataset_2D(
#         data_path=data_path,
#         transform=transform,
#         device=device,
#         output_var=output_var,
#         mode="forecasting_lstm",
#         preprocess=None,
#         variable_to_use=variables_to_use
#     )

#     print("Data collected.")

#     wavelet = getattr(scipy.signal, transform)

#     scales = np.arange(1, 127, 1).astype(int)

#     for i, (sample, label) in tqdm(enumerate(dataset)):
#         signal, image = sample
#         signal = signal.squeeze().numpy()
#         image = image.squeeze().numpy()
#         print(f'Shapes: signal: {signal.shape}, image: {image.shape}')
#         for chan in range(len(variables_to_use)):
#             cv2.imwrite("tmp2.png", image[chan])
#             sig2cwtimg(signal[chan], wavelet, scales, "tmp.png")
#             img_from_signal = load_img("tmp.png")
#             diff_img = abs(image[chan] - img_from_signal)
#             # save the difference image
#             plt.imshow(abs(diff_img), extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
#                     vmax=abs(diff_img).max(), vmin=-abs(diff_img).max())
#             plt.xticks([])
#             plt.yticks([])
#             plt.savefig("diff.png", bbox_inches='tight',pad_inches=0.0 )
#             plt.clf()

#             if not np.allclose(image[chan], img_from_signal):
#                 print(f'Image {i} (channel {variables_to_use[chan]}) is different, max diff: {abs(image - img_from_signal).max()}')
#                 break
#         else:
#             continue
#         print('Some image is different')
#         break
#     else:
#         print('All images are equal')


# def main(data_path, transform, output_var, variables_to_use):

#     batch_size = 1

#     train_val_split = 0.1

#     train_test_split = 0.2

#     mode = "forecasting_lstm"

#     # TODO check if CPU is more accurate
#     device = hardware_check()

#     print("Collecting data..")

#     # TODO check shuffle (seems to change at every execution)

#     train_data_loader, val_data_loader, test_data_loader, output_var_idx = collect_data_2D_lstm(data_path=data_path, transform = transform, device = device, output_var= output_var, train_test_split=train_test_split, train_val_split=train_val_split, mode=mode, batch_size=batch_size, variables_to_use=variables_to_use, csv_file="AirQuality.csv")

#     print("Data collected.")

#     wavelet = getattr(scipy.signal, transform)

#     scales = np.arange(1, 127, 1).astype(int)

#     for i, (sample, label) in tqdm(enumerate(train_data_loader)):
#         signal, image = sample
#         signal = signal.squeeze().numpy()
#         image = image.squeeze().numpy()
#         print(f'Shapes: signal: {signal.shape}, image: {image.shape}')
#         for chan in range(len(variables_to_use)):
#             sig2cwtimg(signal[chan], wavelet, scales, "tmp.png")
#             img_from_signal = load_img("tmp.png")
#             diff_img = abs(image[chan] - img_from_signal)
#             # save the difference image
#             plt.imshow(abs(diff_img), extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
#                     vmax=abs(diff_img).max(), vmin=-abs(diff_img).max())
#             plt.xticks([])
#             plt.yticks([])
#             plt.savefig("diff.png", bbox_inches='tight',pad_inches=0.0 )
#             plt.clf()

#             if not np.allclose(image[chan], img_from_signal):
#                 print(f'Image {i} (channel {variables_to_use[chan]}) is different, max diff: {abs(image - img_from_signal).max()}')
#                 break
#         else:
#             continue
#         print('Some image is different')
#         break
#     else:
#         print('All images are equal')


# if __name__ == "__main__":
#     data_path = "datasets/2D"
#     transform = "ricker"
#     output_var = "CO(GT)"
#     # TODO double check
#     variables_to_use = ["CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
#     variables_to_use = list(sorted(variables_to_use))
#     main(data_path, transform, output_var, variables_to_use)

import torch
from torchvision import transforms
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib.pyplot as plt
from datasets import Dataset_2D
import os
from utils import hardware_check
import torchvision.utils as vutils


class CWTAugmentation:

    def __init__(self, rotation_range=5, scale_range=(0.9, 1.1), translation_range=(0.1, 0.1),
                 zoom_range=(0.9, 1.1), brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
                 noise_std=0.05, elastic_alpha=1, elastic_sigma=0.1):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

    def app(self, cwt_image):
        for i in range(cwt_image.size(0)):
            cwt_image[i,:,:] = self.apply_transform(cwt_image[i,:,:])
        
        return cwt_image

    def __call__(self, cwt_image):

        # Random rotation 
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        cwt_image = self.rotate(cwt_image, angle)
        
        # Random scaling
        scale_factor = np.random.uniform(*self.scale_range)
        cwt_image = self.scale(cwt_image, scale_factor)
        
        # Random translation
        translation = (np.random.uniform(-self.translation_range[0], self.translation_range[0]),
                       np.random.uniform(-self.translation_range[1], self.translation_range[1]))
        cwt_image = self.translate(cwt_image, translation)
        
        # Random zoom
        zoom_factor = np.random.uniform(*self.zoom_range)
        cwt_image = self.zoom(cwt_image, zoom_factor)
        
        # # Random brightness and contrast adjustment
        brightness_factor = np.random.uniform(*self.brightness_range)
        contrast_factor = np.random.uniform(*self.contrast_range)
        for i in range(cwt_image.size(0)):   
            cwt_image[i] = self.adjust_brightness_contrast(cwt_image[i].unsqueeze(0), brightness_factor, contrast_factor).squeeze(0)
        
        # Add Gaussian noise
        cwt_image = self.add_noise(cwt_image,self.noise_std )
        
        # Elastic transformations
        cwt_image = self.elastic_transform(cwt_image)
        
        return cwt_image


    def rotate(self, image, angle):
        return transforms.functional.rotate(image, angle)

    def scale(self, image, scale_factor):
        return transforms.functional.affine(image, angle=0, translate=(0, 0), scale=scale_factor, shear=0)

    def translate(self, image, translation):
        return transforms.functional.affine(image, angle=0, translate=translation, scale=1, shear=0)

    def zoom(self, image, zoom_factor):
        return transforms.functional.affine(image, angle=0, translate=(0, 0), scale=zoom_factor, shear=0)

    def adjust_brightness_contrast(self, image, brightness_factor, contrast_factor):
        return transforms.functional.adjust_brightness(transforms.functional.adjust_contrast(image, contrast_factor),
                                                      brightness_factor)

    def add_noise(self, image, noise_std):
        noise = torch.randn_like(image) * noise_std
        return image + noise

    def elastic_transform(self, image):
        alpha = image.size(1) * self.elastic_alpha
        sigma = image.size(1) * self.elastic_sigma

        random_state = np.random.RandomState(None)
        shape = image.size()
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

        distorted_image = map_coordinates(image.numpy(), indices, order=1, mode='reflect')
        distorted_image = distorted_image.reshape(image.size())

        return torch.from_numpy(distorted_image)


def show_12dim_sample(cwt_image):
    # Reshape the cwt_image to have shape (12, 1, H, W)
    cwt_image = cwt_image.view(12, 1, cwt_image.shape[1], cwt_image.shape[2])

    # Create a grid of the 12 channels
    grid_image = vutils.make_grid(cwt_image, nrow=4, padding=2, normalize=True)

    # Convert the grid image to numpy array
    grid_image_np = grid_image.numpy()

    # Transpose the dimensions to match the channel order
    grid_image_np = np.transpose(grid_image_np, (1, 2, 0))

    # Display the grid image
    plt.imshow(grid_image_np)
    plt.axis('off')
    plt.show()



def save_and_return_one_12dim_sample(image_name):

    data_path = 'datasets/2D'
    output_var = 'CO(GT)'
    transform = 'ricker'
    mode = 'forecasting_lstm'
    device = 'cpu'
    preprocess = None
    variables_to_use = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)',
                        'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 
                        'PT08.S5(O3)', 'T', 'RH', 'AH']

    device = 0 if torch.cuda.is_available() else 'cpu'
    device = hardware_check()

    os.environ["CUDA_VISIBLE_DEVICES"] = device

    print("GPU IN USO: ", device)

    dataset_cwt = Dataset_2D(data_path=data_path, transform=transform, device=device,
                             output_var=output_var, mode=mode, preprocess=preprocess, 
                             variable_to_use=variables_to_use)

    print(f'\nNumero di Training samples: {len(dataset_cwt)}')

    dataset_cwt = iter(dataset_cwt)

    cwt_image = next(dataset_cwt)[0]

    print(cwt_image.shape)

    # Save cwt_image
    torch.save(cwt_image,image_name)

    return cwt_image




if __name__ == "__main__":

    cwt_image =  save_and_return_one_12dim_sample('cwt_images.pt')  ## uncomment to return AND save a sample cwt_image ##

    #cwt_image = torch.load('cwt_images.pt')

    #show_12dim_sample(cwt_image)

    # # Load cwt_image
    # cwt_image = torch.load('cwt_image.pt').unsqueeze(1)

    # # Print the shape of cwt_image
    # print(cwt_image.shape)

    # plt.imshow(cwt_image)
    # plt.show()

    #cwt_image = torch.randn(12, 369, 496) #torch.Size([1, 369, 496]) funziona

    print("cwt_image.shape: ", cwt_image.shape)

    angle = np.random.uniform(-2, 2)
    cwt_image = transforms.functional.rotate(cwt_image, angle)

    #show_12dim_sample(cwt_image)

    transform = CWTAugmentation()

    for i in range(2): augmented_image = transform(cwt_image)

    show_12dim_sample(augmented_image.squeeze(1))



"""
time delle funzioni di augmentation eseguite

os time posix.times_result(user=91.38, system=70.34, children_user=0.0, children_system=0.08, elapsed=17195364.72) 1
os time posix.times_result(user=91.67, system=70.34, children_user=0.0, children_system=0.08, elapsed=17195364.74) 2
os time posix.times_result(user=92.0, system=70.34, children_user=0.0, children_system=0.08, elapsed=17195364.77) 3
os time posix.times_result(user=92.34, system=70.34, children_user=0.0, children_system=0.08, elapsed=17195364.79) 4
os time posix.times_result(user=92.67, system=70.34, children_user=0.0, children_system=0.08, elapsed=17195364.82) 5
os time posix.times_result(user=92.89, system=70.34, children_user=0.0, children_system=0.08, elapsed=17195364.83) 6
os time posix.times_result(user=93.78, system=70.34, children_user=0.0, children_system=0.08, elapsed=17195364.91) 7
os time posix.times_result(user=97.19, system=70.34, children_user=0.0, children_system=0.08, elapsed=17195367.24) 8

"""
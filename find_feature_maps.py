
import torchvision
from utils import *
from StackedResnet import StackedResNet
from torchsummary import summary
from datasets import *

import torch

# Load a pre-trained ResNet model

model = torchvision.models.resnet18(pretrained=True, progress=True)

num_input_channels = 12  # Number of stacked images in input 

model = StackedResNet(num_input_channels, num_output_features=24, resnet=model)

# Remove the fully connected layers (classification layers) from the model
#model = torch.nn.Sequential(*(list(model.children())[:-2]))

# Set the model to evaluation mode
model.eval()

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

# Example usage
if __name__ == "__main__":
    # Load an example image
    image = torch.rand(8, 12, 396, 496)  # Replace with your image dimensions

    # Specify the layer from which you want to extract feature maps
    target_layer = list(model.resnet.children())[-3][1].conv1 #model[-1][1]

    # Get the feature map from the specified layer
    feature_map = get_interested_feature_map(model, image, target_layer)

    print("Feature map shape:", feature_map.shape)
    
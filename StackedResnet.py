import torch.nn as nn

class StackedResNet(nn.Module):

    def __init__(self, num_input_channels, resnet):

        super(StackedResNet, self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, 6, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(6)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(3)

        self.resnet = resnet
        self.resnet.fc = nn.Linear(512, 1)  # last fully connected layer

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.resnet(x)

        return x





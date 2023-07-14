import torch.nn as nn

class Stacked2DLinear(nn.Module):

    def __init__(self, num_input_channels, mode):

        super(Stacked2DLinear, self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, 6, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(6)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(3)
        self.flatten = nn.Flatten()
        if (mode == 'regression'):
            self.fc1 = nn.Linear(5544, 1024) 
        else:
            self.fc1 = nn.Linear(6048, 1024) 
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    


class Stacked1DLinear(nn.Module):

    def __init__(self, num_input_channels, mode):

        super(Stacked1DLinear, self).__init__()


        self.conv1 = nn.Conv1d(num_input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(64*168, 1024)
        self.relu4 = nn.ReLU()
        
        self.linear2 = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = x.squeeze(1) # remove the channel created for the 2D -> [batch, num_parameters, window_size]
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu4(self.linear1(x))
        x = self.linear2(x)
        
        return x





import torch.nn as nn
import torch

class StackedResNet(nn.Module):

    def __init__(self, num_input_channels, num_output_features, resnet):

        super(StackedResNet, self).__init__()

        self.num_output_features = num_output_features

        self.conv1 = nn.Conv2d(num_input_channels, 6, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(6)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(3)

        self.resnet = resnet
        self.resnet.fc = nn.Linear(512, num_output_features)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.resnet(x)

        return x


# opzioni:
# 1. copiare tutte le feature per ogni timestep
# 2. reshape delle feature da (N,) a (N/T, T) con T lunghezza finestra
# 3. fully connected che porta le feature da (N,) a (T,)
# 4. baseline con StackedResNet con 24 output

class LSTMForecaster(nn.Module):

    def __init__(
        self,
        stacked_resnet,
        channels=11,
        num_layers=2,
        hidden_size=512,
        outputs=1,
        mode='option1'
    ) -> None:
        super().__init__()

        assert mode in ['option1']

        self.stacked_resnet = stacked_resnet

        self.lstm = nn.LSTM(
            input_size=channels + stacked_resnet.num_output_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, outputs)

    # x_img: (N, H, W, C)
    # x_signal: (N, L, C)
    def forward(self, x_img, x_signal):

        L = x_signal.shape[1]
        # (N, F)
        features = self.stacked_resnet(x_img)
        # (N, L, C+F)
        x_lstm = torch.cat(
            (features.unsqueeze(1).expand(-1, L, -1), x_signal),
            dim=2
        )
        # (N, L, H)
        y_lstm, _ = self.lstm(x_lstm)
        # (N, H)
        y_lstm = y_lstm[:, -1, :].squeeze()
        # (N, H/2)
        y_fc1 = self.relu(self.fc1(y_lstm))
        # (N, Y)
        y_fc2 = self.relu(self.fc2(y_fc1))

        return y_fc2



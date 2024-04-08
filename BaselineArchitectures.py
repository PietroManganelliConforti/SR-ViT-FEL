import torch.nn as nn

"""
==========================================================================================
Layer (type:depth-idx)               	Output Shape          	Param #
==========================================================================================
├─Conv2d: 1-1                        	[-1, 6, 12, 168]      	60
├─BatchNorm2d: 1-2                   	[-1, 6, 12, 168]      	12
├─ReLU: 1-3                          	[-1, 6, 12, 168]      	--
├─Conv2d: 1-4                        	[-1, 3, 12, 168]      	165
├─BatchNorm2d: 1-5                   	[-1, 3, 12, 168]      	6
├─ReLU: 1-6                          	[-1, 3, 12, 168]      	--
├─Flatten: 1-7                       	[-1, 6048]            	--
├─Linear: 1-8                        	[-1, 1024]            	6,194,176
├─ReLU: 1-9                          	[-1, 1024]            	--
├─Linear: 1-10                       	[-1, 1]               	1,025
==========================================================================================
Total params: 6,195,444
Trainable params: 6,195,444
Non-trainable params: 0
Total mult-adds (M): 6.63
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.28
Params size (MB): 23.63
Estimated Total Size (MB): 23.93
==========================================================================================
"""

class Stacked2DLinear(nn.Module):

    def __init__(self, num_input_channels, mode):

        super(Stacked2DLinear, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(6)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(3)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3*num_input_channels*168, 1024) 
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
    


"""
==========================================================================================
Layer (type:depth-idx)               	Output Shape          	Param #
==========================================================================================
├─Conv1d: 1-1                        	[-1, 32, 168]         	1,184
├─BatchNorm1d: 1-2                   	[-1, 32, 168]         	64
├─ReLU: 1-3                          	[-1, 32, 168]         	--
├─Conv1d: 1-4                        	[-1, 64, 168]         	6,208
├─BatchNorm1d: 1-5                   	[-1, 64, 168]         	128
├─ReLU: 1-6                          	[-1, 64, 168]         	--
├─Flatten: 1-7                       	[-1, 10752]           	--
├─Linear: 1-8                        	[-1, 1024]            	11,011,072
├─ReLU: 1-9                          	[-1, 1024]            	--
├─Linear: 1-10                       	[-1, 1]               	1,025
==========================================================================================
Total params: 11,019,681
Trainable params: 11,019,681
Non-trainable params: 0
Total mult-adds (M): 12.24
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.25
Params size (MB): 42.04
Estimated Total Size (MB): 42.30
==========================================================================================
"""



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


"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─LSTM: 1-1                              [-1, 1024, 1024]          12,648,448
├─Linear: 1-2                            [-1, 1024]                1,049,600
├─ReLU: 1-3                              [-1, 1024]                --
├─Linear: 1-4                            [-1, 1]                   1,025
==========================================================================================
Total params: 13,699,073
Trainable params: 13,699,073
Non-trainable params: 0
Total mult-adds (M): 13.68
==========================================================================================
Input size (MB): 0.05
Forward/backward pass size (MB): 8.01
Params size (MB): 52.26
Estimated Total Size (MB): 60.31
==========================================================================================
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─LSTM: 1-1                              [-1, 1024, 1024]          12,648,448
├─Linear: 1-2                            [-1, 1024]                1,049,600
├─ReLU: 1-3                              [-1, 1024]                --
├─Linear: 1-4                            [-1, 1]                   1,025
==========================================================================================
Total params: 13,699,073
Trainable params: 13,699,073
Non-trainable params: 0
Total mult-adds (M): 13.68
==========================================================================================
Input size (MB): 0.05
Forward/backward pass size (MB): 8.01
Params size (MB): 52.26
Estimated Total Size (MB): 60.31
==========================================================================================
"""


class LSTMLinear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMLinear, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(hidden_size, 1024)
        self.relu1 = nn.ReLU()
        
        self.linear2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.squeeze(1)
        # x has shape (batch, channels, length), we need to rearrange it to (batch, length, channels)
        x = x.permute(0, 2, 1)
        lstm_output, _ = self.lstm(x)
        # Take the last output of the LSTM sequence (corresponding to the last time step)
        lstm_output = lstm_output[:, -1, :]
        
        x = self.relu1(self.linear1(lstm_output))
        x = self.linear2(x)
        
        return x
    
    

class LSTMForecaster(nn.Module):

    def __init__(
        self,
        outputs,
        channels,
        num_layers,
        hidden_size,
        bidirectional,
        mode='option1'
    ) -> None:
        super().__init__()

        assert mode in ['option1']

        input_size = channels
        print(f'channels: {channels},input_size: {input_size}, outputs: {outputs}')
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear((2 if bidirectional else 1)*hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, outputs)

    # x_img: (N, C, H, W)
    # x_signal: (N, L, C)
    # -> (N, Y)
    def forward(self, x_signal):
        # print(f'Shapes: x_img: {x_img.shape}, x_sig: {x_signal.shape}')
        L = x_signal.shape[1]
        
        x_lstm = x_signal
        
        # (N, L, H)
        y_lstm, _ = self.lstm(x_lstm)
        # (N, H)
        y_lstm = y_lstm[:, -1, :].squeeze()
        # (N, H/2)
        y_fc1 = self.relu(self.fc1(y_lstm))
        # (N, Y)
        y_fc2 = self.relu(self.fc2(y_fc1))

        return y_fc2
    
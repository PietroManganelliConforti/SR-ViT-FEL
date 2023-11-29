import torch.nn as nn
import torch.nn.functional as F
import torch
import transformers
from transformers import ViTModel
from utils import get_interested_feature_map

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
        # TODO remove this layer
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

        input_size = channels + stacked_resnet.num_output_features
        print(f'channels: {channels}, stacked_resnet.num_output_features: {stacked_resnet.num_output_features}, input_size: {input_size}')
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, outputs)

    # x_img: (N, H, W, C)
    # x_signal: (N, L, C)
    # -> (N, Y)
    def forward(self, x_img, x_signal):

        L = x_signal.shape[2]
        # (N, F)
        features = self.stacked_resnet(x_img)
        # (N, L, C+F)
        print('AO' + str(features.unsqueeze(1).expand(-1, L, -1).shape) + str(x_signal.shape))
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


# https://github.com/ruiqiRichard/EEGViT/blob/master/models/EEGViT_pretrained.py
"""
ViTForImageClassification(
  (vit): ViTModel(
    (embeddings): ViTEmbeddings(
      (patch_embeddings): ViTPatchEmbeddings(
        (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
      )
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (encoder): ViTEncoder(
      (layer): ModuleList(
        (0-11): 12 x ViTLayer(
          (attention): ViTAttention(
            (attention): ViTSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): ViTSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): ViTIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ViTOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  )
  (classifier): Linear(in_features=768, out_features=1000, bias=True)
)
"""

# 1. 512 features from stackedresnet into encoder
# 2. feature maps from stackedresent as images for ViT
# 3. transforms as images for ViT
class ViTForecaster (nn.Module):
    def __init__(
        self,
        stacked_resnet,
        dim,
        #channels=11,
        #hidden_size=512,
        outputs=24,
    ) -> None:
        super().__init__()
        self.conv1x1 = torch.nn.Conv2d(64, 3, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(3)

        self.dim = dim
        if (self.dim == '2D_ViT_im'):
            self.conv1x1 = torch.nn.Conv2d(12, 3, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(3)
        
        elif (self.dim == '2D_ViT_feat'):
            self.conv1x1 = torch.nn.Conv2d(64, 3, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(3)
            self.stacked_resnet = stacked_resnet
            self.stacked_resnet.resnet.fc = nn.Identity()
        


        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        
        # TODO: can we use this config update to avoid interpolation?
        # config.update({'num_channels': 256})
        # config.update({'image_size': (129,14)})
        # config.update({'patch_size': (8,1)})
        
        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)        
        #model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(512, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0,0), groups=256)
        model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
                                       torch.nn.Dropout(p=0.1),
                                       torch.nn.Linear(1000,outputs,bias=True))
        self.ViT = model

    # x_img: (N, H, W, C)
    # -> (N, Y)
    def forward(self, x_img):
        
        if (self.dim == '2D_ViT_im'):
            # From (8, 12, 396, 496) to (8, 3, 224, 224) 
            x_img = F.interpolate(x_img, size=(224, 224), mode='bilinear', align_corners=False)
            x_img = self.conv1x1(x_img)
            x = self.bn1(x_img)
            
        elif (self.dim == '2D_ViT_feat'):
            feature_map = get_interested_feature_map(self.stacked_resnet, x_img, list(self.stacked_resnet.resnet.children())[-6][1].conv1)
        
            # From (8, 12, 396, 496) to (8, 3, 224, 224) 
            feature_map = F.interpolate(feature_map, size=(224, 224), mode='bilinear', align_corners=False)
            feature_map = self.conv1x1(feature_map)
            x = self.bn1(feature_map)
            
        outputs = self.ViT(x).logits

        return outputs


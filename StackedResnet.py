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



class AdaptaBlock(nn.Module):
    def __init__(self, UPCONV=False):
        super(AdaptaBlock, self).__init__()

        
        # 512 -> 3
        self.conv00 = torch.nn.Conv2d(512, 256, kernel_size=1)
        self.bn00 = nn.BatchNorm2d(256)
        self.conv01 = torch.nn.Conv2d(256, 128, kernel_size=1)
        self.bn01 = nn.BatchNorm2d(128)
        self.conv02 = torch.nn.Conv2d(128, 64, kernel_size=1)
        self.bn02 = nn.BatchNorm2d(64)
        self.conv03 = torch.nn.Conv2d(64, 32, kernel_size=1)
        self.bn03 = nn.BatchNorm2d(32)
        self.conv04 = torch.nn.Conv2d(32, 1, kernel_size=1)
        self.bn04 = nn.BatchNorm2d(1) # 12, 16

        # Define ConvTranspose2d and BatchNorm2d layers
        self.conv_transpose00 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1) # 23, 31
        self.bnt00 = nn.BatchNorm2d(256)
        self.conv_transpose01 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1) # 45, 61
        self.bnt01 = nn.BatchNorm2d(128)
        self.conv_transpose02 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1) # 89, 121
        self.bnt02 = nn.BatchNorm2d(64)
        self.conv_transpose03 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1) # 177, 241
        self.bnt03 = nn.BatchNorm2d(32)


        # 256 -> 3
        self.conv10 = torch.nn.Conv2d(256, 128, kernel_size=1)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = torch.nn.Conv2d(128, 64, kernel_size=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = torch.nn.Conv2d(64, 32, kernel_size=1)
        self.bn12 = nn.BatchNorm2d(32)
        self.conv13 = torch.nn.Conv2d(32, 1, kernel_size=1)
        self.bn13 = nn.BatchNorm2d(1) # 24, 31

        self.conv_transpose10 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1) # 47, 61
        self.bnt10 = nn.BatchNorm2d(128)
        self.conv_transpose11 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1) # 93, 121
        self.bnt11 = nn.BatchNorm2d(64)
        self.conv_transpose12 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1) # 185, 241
        self.bnt12 = nn.BatchNorm2d(32)

        # 128 -> 3
        self.conv20 = torch.nn.Conv2d(128, 64, kernel_size=1)
        self.bn20 = nn.BatchNorm2d(64)
        self.conv21 = torch.nn.Conv2d(64, 32, kernel_size=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.conv22 = torch.nn.Conv2d(32, 1, kernel_size=1)
        self.bn22 = nn.BatchNorm2d(1) # 47, 62 

        self.conv_transpose20 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1) # 93, 123
        self.bnt20 = nn.BatchNorm2d(64)
        self.conv_transpose21 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1) # 185, 245
        self.bnt21 = nn.BatchNorm2d(32)
        
        self.relu = nn.ReLU()

    def forward (self, tensor0, tensor1, tensor2):
        # 8, 512 , 12, 16   
        x = self.relu(self.bn00(self.conv00(tensor0)))
        x = self.bnt00(self.conv_transpose00(x))
        x = self.relu(self.bn01(self.conv01(x)))
        x = self.bnt01(self.conv_transpose01(x))
        x = self.relu(self.bn02(self.conv02(x)))
        x = self.bnt02(self.conv_transpose02(x))
        x = self.relu(self.bn03(self.conv03(x)))
        x = self.bnt03(self.conv_transpose03(x))
        x_feat_0 = self.relu(self.bn04(F.interpolate(self.conv04(x), size=(224, 224), mode='bilinear', align_corners=False))) # 8, 1, 224, 224   
        
        # 8, 256 , 24, 31
        x = self.relu(self.bn10(self.conv10(tensor1)))
        x = self.bnt10(self.conv_transpose10(x))
        x = self.relu(self.bn11(self.conv11(x)))
        x = self.bnt11(self.conv_transpose11(x))
        x = self.relu(self.bn12(self.conv12(x)))
        x = self.bnt12(self.conv_transpose12(x))
        x_feat_1 = self.relu(self.bn13(F.interpolate(self.conv13(x), size=(224, 224), mode='bilinear', align_corners=False))) # 8, 1, 224, 224

        # 8 , 128 , 47, 62 
        x = self.relu(self.bn20(self.conv20(tensor2)))
        x = self.bnt20(self.conv_transpose20(x))
        x = self.relu(self.bn21(self.conv21(x)))
        x = self.bnt21(self.conv_transpose21(x))
        x_feat_2 = self.relu(self.bn22(F.interpolate(self.conv22(x), size=(224, 224), mode='bilinear', align_corners=False))) # 8, 1, 224, 224

        
        return torch.cat((x_feat_0,x_feat_1, x_feat_2), axis=1)




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

        self.dim = dim
        
        
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        
        # TODO: can we use this config update to avoid interpolation?
        #config.update({'num_channels': 12})
        # config.update({'image_size': (396,496)})
        # config.update({'patch_size': (8,1)})
        
        self.ViT = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)        
        #model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(512, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0,0), groups=256)
        self.ViT.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
                                       torch.nn.Dropout(p=0.1),
                                       torch.nn.Linear(1000,outputs,bias=True))        
        
        self.relu = nn.ReLU()
        if (self.dim == '2D_ViT_im'):
            self.conv1x1 = torch.nn.Conv2d(12, 3, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(3)
        

            
        elif (self.dim == '2D_ViT_parallel_SR'):
            self.stacked_resnet = stacked_resnet
            self.stacked_resnet.resnet.fc = nn.Identity()
            
            self.conv1x1 = torch.nn.Conv2d(12, 3, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(3)
            self.ViT.classifier = torch.nn.Sequential(torch.nn.Linear(768,512,bias=True),torch.nn.Dropout(p=0.1), torch.nn.ReLU())
            
            self.classifier = torch.nn.Sequential(torch.nn.Linear(512*2,24,bias=True))
    
        
        elif (self.dim == '2D_ViT_SR_feat_in'):
            self.stacked_resnet = stacked_resnet
            self.stacked_resnet.resnet.fc = nn.Identity()
            self.adapta_block = AdaptaBlock()
            
            self.ViT.classifier = torch.nn.Sequential(torch.nn.Linear(768,512,bias=True),torch.nn.Dropout(p=0.1), torch.nn.ReLU())
              
            self.classifier = torch.nn.Sequential(torch.nn.Linear(512*2,24,bias=True))
            

    # x_img: (N, H, W, C)
    # -> (N, Y)
    def forward(self, x_img):
        
        if (self.dim == '2D_ViT_im'):
            x = self.conv1x1(x_img)
            x = self.bn1(x)
            # From (8, 12, 396, 496) to (8, 3, 224, 224) 
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            outputs = self.ViT(x, interpolate_pos_encoding=True).logits
           
        elif (self.dim == '2D_ViT_parallel_SR'):
            x_sr = self.stacked_resnet(x_img)
            
            x_img = F.interpolate(x_img, size=(224, 224), mode='bilinear', align_corners=False)
            x_img = self.conv1x1(x_img)
            x_img = self.bn1(x_img)
            x_vit = self.ViT(x_img).logits
            
            
            x = torch.cat((x_sr, x_vit), 1)
            outputs = self.classifier(x)

            
        elif (self.dim == '2D_ViT_SR_feat_in'):
            x_sr = self.stacked_resnet(x_img)
              
                
            tensor0 = get_interested_feature_map(self.stacked_resnet, x_img, list(self.stacked_resnet.resnet.children())[-3][-1].conv2) # 8 512 12 16
            tensor1 = get_interested_feature_map(self.stacked_resnet, x_img, list(self.stacked_resnet.resnet.children())[-4][-1].conv2) # 8 256 24 31
            tensor2 = get_interested_feature_map(self.stacked_resnet, x_img, list(self.stacked_resnet.resnet.children())[-5][-1].conv2) # 8 128 47 62

            x_feat_merged = self.adapta_block(tensor0, tensor1, tensor2)
            x_vit = self.ViT(x_feat_merged).logits
              
            x = torch.cat((x_sr, x_vit), 1)
            outputs = self.classifier(x)            
        
        return outputs


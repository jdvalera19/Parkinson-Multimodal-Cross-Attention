import torch

import torch.nn as nn

class CNNModel2D(nn.Module):
    def __init__(self):
        super(CNNModel2D, self).__init__()

        #self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer1 = self._conv_layer_set(2, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        #self.fc1 = nn.Linear(373248, 128) #3D
        self.fc1 = nn.Linear(194432, 128) #2D Vowels
        #self.fc1 = nn.Linear(154752, 128) #2D Phonemes
        #self.fc1 = nn.Linear(59520, 128) #2D Words
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        #nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0), #3D Convolution
        nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=0), #2D Convolution
        nn.LeakyReLU(),
        #nn.MaxPool3d((2, 2, 2)), #3D Convolution
        nn.MaxPool2d((2, 2)), #2D Convolution
        )
        return conv_layer

    def forward(self, x, return_features=False, return_embedding=False):
        # Set 1
        #print(x.shape)
        out = self.conv_layer1(x)
        #print(f"After conv_layer1: {out.shape}")
        out = self.conv_layer2(out)
        #out = self.conv1x1(out)   
        #out = self.conv_layer3(out)
        out = out.view(out.size(0), -1)
        """
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        out = self.conv_layer6(out)
        out = self.conv_layer7(out)
        out = self.conv_layer8(out)
        out = self.conv_layer9(out)
        """
        if return_features:
            return out        
        #print(out.shape)
        out = self.fc1(out)
        #print(out.shape)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        if return_embedding:
            return out        
        out = self.fc2(out)
        
        return out

#model = CNNModel()
#model(torch.randn(5 , 1, 224, 224)) #2d input tensor
#model(torch.randn(5 , 1, 16, 224, 224)) #3d input tensor 


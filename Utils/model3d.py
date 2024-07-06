import torch

import torch.nn as nn

class CNNModel3D(nn.Module):
    def __init__(self):
        super(CNNModel3D, self).__init__()

        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)      
        #¿Aquí conv 1x1?
        self.conv1x1 = nn.Conv3d(64, 28, kernel_size=(1, 1, 1))  # 1x1 Convolution que sea solo 1 característica       
        #self.fc1 = nn.Linear(1679616, 128)
        self.fc1 = nn.Linear(734832, 4096) #Hay que calcular este numero
        #self.fc1 = nn.Linear(1306368, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x, return_features=False):
        # Set 1
        #print(x.shape)
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv1x1(out)        
        out = out.view(out.size(0), out.size(1)*out.size(2), out.size(3), out.size(4))
        #print(out.shape)
        if return_features:
            return out          
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out
    

    def get_embedding(self, x):
        with torch.no_grad():
            out = self.conv_layer1(x)
            out = self.conv_layer2(out)
            out = self.conv1x1(out)        
            out = out.view(out.size(0), out.size(1) * out.size(2) * out.size(3) * out.size(4))
            out = out.view(out.size(0), -1) #aplanar tensor para el embebido
            print(out.size())
            embedding = self.fc1(out)
        return embedding    



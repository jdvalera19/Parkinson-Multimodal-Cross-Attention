import torch.nn as nn
from Utils.MultiHeadAttn_V2 import RFBMultiHeadAttn_V2
import torch as T
class RFBMultiHAttnNetwork_V3(nn.Module):
    def __init__(self, in_dim1, in_dim2):
        super().__init__()
        self.heads = 3
        self.RFBMHA_V3 = RFBMultiHeadAttn_V2(in_dim1, 32, self.heads)  
        self.RFBMHA_V4 = RFBMultiHeadAttn_V2(in_dim2, 32, self.heads)  
        self.fc1 = nn.Linear(32*self.heads*2, 16)  
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x1, x2):
        x1 = self.RFBMHA_V3(x1)
        x2 = self.RFBMHA_V4(x2)
        x = T.cat((x1, x2), dim=1)  
        x = x.view(-1, x.size(1))
        x = self.fc1(x)
        x = T.sigmoid(x)
        return x
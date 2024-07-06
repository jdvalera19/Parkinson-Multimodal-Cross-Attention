import torch.nn as nn
import torch as T
from Utils.BasicConv3dRBF import BasicConv3dRBF
class RBF_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RBF_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv3dRBF(in_channel, out_channel, (1, 1, 1)),
        )
        self.branch1 = nn.Sequential(
            BasicConv3dRBF(in_channel, out_channel, (1, 1, 1)),
            BasicConv3dRBF(out_channel, out_channel, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
            BasicConv3dRBF(out_channel, out_channel, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            BasicConv3dRBF(out_channel, out_channel, (3, 3, 3), padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv3dRBF(in_channel, out_channel, (1, 1, 1)),
            BasicConv3dRBF(out_channel, out_channel, kernel_size=(1, 5, 1), padding=(0, 2, 0)),
            BasicConv3dRBF(out_channel, out_channel, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            BasicConv3dRBF(out_channel, out_channel, (3, 3, 3), padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv3dRBF(in_channel, out_channel, (1, 1, 1)),
            BasicConv3dRBF(out_channel, out_channel, kernel_size=(1, 7, 1), padding=(0, 3, 0)),
            BasicConv3dRBF(out_channel, out_channel, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            BasicConv3dRBF(out_channel, out_channel, (3, 3, 3), padding=7, dilation=7)
        )
        self.conv_cat = BasicConv3dRBF(4*out_channel, out_channel, (3, 3, 3), padding=1)
        self.conv_res = BasicConv3dRBF(in_channel, out_channel, (1, 1, 1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(T.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        
        return x
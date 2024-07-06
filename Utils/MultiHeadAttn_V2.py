import torch.nn as nn
from einops.layers.torch import Rearrange
import torch as T
from Utils.RBF_modified3d import RBF_modified3d
from Utils.RBF_modified2d import RBF_modified2d
class RFBMultiHeadAttn_V2(nn.Module):
    def __init__(self, in_dim, filters_head, num_multiheads):
        super(RFBMultiHeadAttn_V2, self).__init__()  
        """
            Multi-head attention
            in_dim = entry dimmension
            filters_head = filters by head
            We plan to make as much attentions as possible, in order to provide a reliable
            classification. Thus, creating different types of projections called k, q and v.
        """    
        self.in_dim = in_dim
        self.filters_head = filters_head
        self.num_multiheads = num_multiheads
        self.inner_filters = filters_head * num_multiheads

        #Queries of each attention made
        self.qkv_rfb = RBF_modified2d(in_dim, self.inner_filters * 3)

        self.rearrange_for_matmul = Rearrange(
            #"b c (d nh) h w  -> b c nh d h w", nh=num_multiheads
            "b (nh c) d h w  -> b nh c d h w", nh=num_multiheads
        )
        self.rearrange_back = Rearrange("b nh c d h w -> b (c nh) d h w")
        self.rearrange_for = Rearrange("nh b c d h w -> b (c nh) d h w")
        self.gamma_one = nn.Parameter(T.zeros(1))
        self.gamma_two = nn.Parameter(T.zeros(1))
        self.gamma_thr = nn.Parameter(T.zeros(1))
        self.gamma_fou = nn.Parameter(T.zeros(1))
        self.gamma_fiv = nn.Parameter(T.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X D X W X H)
            returns :
                out : self attention value + input feature + num of heads
                attention: B X C X D X N (N is Width*Height)
        """
        
        #It is important to obtain a size of the input
        m_batchsize, C, dim, width, height = x.size()
#         print("x:        ", x.shape)
        #Now, we're gonna obtain an attention for each one 
#         print(self.qkv_rfb(x).shape)
        proj_qkv = self.qkv_rfb(x).view(m_batchsize, -1, dim, width, height)
#         print("proj_qkv: ", proj_qkv.shape)
        proj_qkv_rearranged = self.rearrange_for_matmul(proj_qkv)
#         print("rearanfed:", proj_qkv_rearranged.shape)
        q, k, v = proj_qkv_rearranged.chunk(chunks=3, dim=2)
#         print(q.shape, k.shape, v.shape)
        #return v para sonido #PARTE IMPORTANTE

        sim = (k @ q.permute(0, 1, 2, 3, 5, 4))
        att_map = self.softmax(sim)  # BX (N) X (N)
        proj_v = att_map @ v 
#         print("proj_v:   ", proj_v.shape)
        out_att = self.rearrange_back(proj_v)
#         print("out_att:  ", out_att.shape)
        out_heads = out_att.chunk(chunks=self.num_multiheads, dim=1)
#         print("out_heads:", out_heads[0].shape, out_heads[1].shape, out_heads[2].shape)
        out_gamma = T.cat([self.gamma_one,self.gamma_two,self.gamma_thr,self.gamma_fou,self.gamma_fiv])
        result = T.stack([(out_gamma[index] * heads + x) for index, heads in enumerate(out_heads)])
#         print("result;   ", result.shape)
        return self.rearrange_for(result)
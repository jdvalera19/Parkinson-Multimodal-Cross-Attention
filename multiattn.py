import math
import os
from einops import rearrange
import torch



""" os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"]   = "1"
 """
import torch as T
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F 


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv3x3_3d(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3_3d(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_3d(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class BasicConv2dRBF(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2dRBF, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # Qué pasa si le agrego un ReLU aquí?
        # x = self.relu(x)
        return x
    
class RBF_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RBF_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2dRBF(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2dRBF(in_channel, out_channel, 1),
            BasicConv2dRBF(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2dRBF(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2dRBF(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2dRBF(in_channel, out_channel, 1),
            BasicConv2dRBF(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2dRBF(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2dRBF(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2dRBF(in_channel, out_channel, 1),
            BasicConv2dRBF(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2dRBF(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2dRBF(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2dRBF(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2dRBF(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(T.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        
        return x

class BasicConv3dRBF(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3dRBF, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class RBF_modified3D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RBF_modified3D, self).__init__()
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
    

class RFBMultiHeadAttn_V1(nn.Module):
    def __init__(self, in_dim, filters_head, num_multiheads):
        super(RFBMultiHeadAttn_V1, self).__init__()  
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
        self.qkv_rfb = RBF_modified3D(in_dim, self.inner_filters * 3)

        self.rearrange_for_matmul = Rearrange(
            #"b (d nh) h w  -> b nh d h w", nh=num_multiheads
            "b (nh d) h w  -> b nh d h w", nh=num_multiheads
        )
        self.rearrange_back = Rearrange("b nh d h w -> b (nh d) h w")
        self.rearrange_for = Rearrange("nh b d h w -> b (nh d) h w")
        self.gamma_one = nn.Parameter(T.zeros(1))
        self.gamma_two = nn.Parameter(T.zeros(1))
        self.gamma_thr = nn.Parameter(T.zeros(1))
        self.gamma_fou = nn.Parameter(T.zeros(1))
        self.gamma_fiv = nn.Parameter(T.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature + num of heads
        """
        #x = x.reshape(5, 8, 4, 4)  
        #It is important to obtain a size of the input
        x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3), x.size(4))
        m_batchsize, C, width, height = x.size()

        #Now, we're gonna obtain an attention for each one 
        proj_qkv = self.qkv_rfb(x).view(m_batchsize, -1, width,  height)
        #print("proj_qkv", proj_qkv.shape)
        proj_qkv_rearranged = self.rearrange_for_matmul(proj_qkv)
        #print(proj_qkv_rearranged.shape)
        q, k, v = proj_qkv_rearranged.chunk(chunks=3, dim=2)

        sim = (k @ q.permute(0, 1, 2, 4, 3))
        att_map = self.softmax(sim)  # BX (N) X (N)
        proj_v = att_map @ v
        
        out_att = self.rearrange_back(proj_v)
        out_heads = out_att.chunk(chunks=self.num_multiheads, dim=1)
        out_gamma = T.cat([self.gamma_one,self.gamma_two,self.gamma_thr,self.gamma_fou,self.gamma_fiv])
        result = T.stack([(out_gamma[index] * heads + x) for index, heads in enumerate(out_heads)])

        return self.rearrange_for(result)


class RFBMultiHAttnNetwork_V3(nn.Module): 
    def __init__(self):
        super().__init__()
        self.heads = 3 #Probar con una sola cabeza
        self.RFBMHA_V3 = nn.Sequential(
            BasicBlock(1, 32), #Video y después audio
            #RFBMultiHeadAttn_V1(32, 32, self.heads),
            RFBMultiHeadAttn_V1(32, 32, self.heads),
            #nn.BatchNorm2d(32*self.heads),
            nn.BatchNorm3d(32*self.heads),
            nn.Dropout(0.1),
#             Basic3dBlock(32*self.heads, 32*self.heads),      
#             RFBMultiHeadAttn_V2(32*self.heads, 32*self.heads, self.heads),
#             nn.BatchNorm3d(32*self.heads*self.heads),
#             nn.Dropout(0.1),
            #nn.AdaptiveAvgPool2d((1,1)),
            nn.AdaptiveAvgPool3d((1,1,1)),
            #PrintLayer()
        )
        self.fc1 = nn.Linear(32*self.heads, 16)
        self.fc2 = nn.Linear(16,2)


    def forward(self, x):
        x = self.RFBMHA_V3(x)
        x = x.view(-1, x.size()[1])
        x = self.fc1(x)
        x = self.fc2(x)
        x = T.sigmoid(x)
        return x 


class RFBMultiHeadAttn_V2(nn.Module):
    def __init__(self, in_dim_q, in_dim_kv, filters_head, num_multiheads, num_classes=2, dropout_rate=0., target_features=4):
        super(RFBMultiHeadAttn_V2, self).__init__()
        
        self.num_multiheads = num_multiheads
        self.filters_head = filters_head
        self.inner_filters = filters_head * num_multiheads

        # Capas para Q, K y V
        self.q_rfb = nn.Linear(in_dim_q, self.inner_filters)
        self.kv_rfb = nn.Linear(in_dim_kv, self.inner_filters * 2)  # K y V juntos

        # Rearrange helper for multi-head attention operation
        self.rearrange_for_matmul = Rearrange('b (nh d) h w -> b nh d h w', nh=num_multiheads)
        self.rearrange_back = Rearrange('b nh d h w -> b (nh d) h w')
        self.softmax = nn.Softmax(dim=-1)  # softmax over last dimension for attention calculation

        # Classification layer
        self.output_linear = nn.Linear(target_features, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.batnorm = nn.BatchNorm1d(target_features)
        self.target_features = target_features
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_features)
        self.relu = nn.LeakyReLU()

        # Gamma parameters for scaling the attention output
        self.gamma = nn.Parameter(torch.zeros(num_multiheads))

    def forward(self, q_input, kv_input):
        current_batch_size = q_input.size(0)
        # Project Q, K, and V
        proj_q = self.q_rfb(q_input).view(-1, self.num_multiheads, self.filters_head)
        proj_kv = self.kv_rfb(kv_input).view(-1, self.num_multiheads, 2 * self.filters_head)

        # Split into K and V
        k, v = proj_kv.chunk(2, dim=2)

        # Attention mechanism
        # Attempt to reshape k using dynamic batch size
        try:
            k = k.reshape(current_batch_size, self.num_multiheads, 1, self.filters_head)
            q = proj_q.reshape(current_batch_size, self.num_multiheads, self.filters_head, 1)
        except RuntimeError as e:
            print("Error during k reshape:", e)
            total_elements = torch.numel(k)
            print("Total elements in k:", total_elements)
        
        
        
        # Attention mechanism
        #k = k.reshape(5, self.num_multiheads, 1, self.filters_head)  # Introduce sequence length of 1
        #q = proj_q.reshape(5, self.num_multiheads, self.filters_head, 1)
        v = v.view(-1, self.num_multiheads, self.filters_head)

        sim = torch.matmul(k, q)
        # Reshape sim to remove singleton dimensions for softmax
        sim = sim.squeeze(-1).squeeze(-1)
        att_map = self.softmax(sim)  # Apply softmax to get attention map
        att_output = torch.matmul(att_map, v)  # Apply attention to V

        # Reshape and project for classification
        combined_heads = att_output.flatten(start_dim=1)
        combined_heads = combined_heads.unsqueeze(1)
        combined_heads = self.adaptive_pool(combined_heads)
        combined_heads = combined_heads.squeeze(1)
        combined_heads = self.relu(combined_heads)
        combined_heads = self.batnorm(combined_heads)
        combined_heads = self.dropout(combined_heads)
        outputs = self.output_linear(combined_heads)
        

        return outputs


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, filters_head):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.filters_head = filters_head

        # Projections for queries, keys, and values
        self.query_conv = RBF_modified(query_dim, self.filters_head) #Quitar RBF_modified para no usar RFB
        self.key_conv = RBF_modified(context_dim, self.filters_head)
        self.value_conv = RBF_modified(context_dim, self.filters_head)
        """
        self.rearrange_for_matmul = Rearrange(
            #"b (d nh) h w  -> b nh d h w", nh=num_multiheads
            "b (nh d) h w  -> b nh d h w", nh=num_multiheads
        )
        self.rearrange_back = Rearrange("b nh d h w -> b (nh d) h w")
        self.rearrange_for = Rearrange("nh b d h w -> b (nh d) h w")
        self.gamma_one = nn.Parameter(T.zeros(1))
        self.gamma_two = nn.Parameter(T.zeros(1))
        self.gamma_thr = nn.Parameter(T.zeros(1))
        self.gamma_fou = nn.Parameter(T.zeros(1))
        self.gamma_fiv = nn.Parameter(T.zeros(1))
        """
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_input, context_input):
        #m_batchsize_query, C_q, width_q, height_q = query_input.size()
        m_batchsize_query, C_q = query_input.size()
        #m_batchsize_context, C_c, width_c, height_c = context_input.size() #REVISAR POR QUE NO SE USA W y H
        m_batchsize_context, C_c = context_input.size()

        # Se proyecta el q, k y v
        #proj_q = self.query_conv(query_input).view(m_batchsize_query, -1, width_q * height_q)
        proj_q = self.query_conv(query_input)
        #proj_k = self.key_conv(context_input).view(m_batchsize_context, -1, width_c * height_c)
        proj_k = self.key_conv(context_input)
        #proj_v = self.value_conv(context_input).view(m_batchsize_context, -1, width_c * height_c) #REVISAR ESTA TRANSPOSICIÓN
        proj_v = self.value_conv(context_input)

        proj_k = proj_k.transpose(-2, -1)

        # Se calcula la matriz de atención
        att_scores = T.matmul(proj_q, proj_k) / math.sqrt(proj_q.size(-1))
        att_map = self.softmax(att_scores)  

        # Se pondera el v con la matriz de atención
        att_output = T.matmul(att_map, proj_v)  

        #att_output = att_output.view(m_batchsize_query, -1, width_q, height_q)

        return att_output

class CrossAttentionEmbedding(nn.Module):
    def __init__(self, query_dim, context_dim, filters_head):
        super(CrossAttentionEmbedding, self).__init__()
        self.query_fc = nn.Linear(query_dim, filters_head)
        self.key_fc = nn.Linear(context_dim, filters_head)
        self.value_fc = nn.Linear(context_dim, filters_head)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_input, context_input):
        # query_input: [batch_size, feature_dim]
        # context_input: [batch_size, feature_dim]

        m_batchsize_query, C_q = query_input.size()
        m_batchsize_context, C_c = context_input.size()

        # Proyectar q, k, y v
        proj_q = self.query_fc(query_input)  # [batch_size, hidden_dim]
        proj_k = self.key_fc(context_input)  # [batch_size, hidden_dim]
        proj_v = self.value_fc(context_input)  # [batch_size, hidden_dim]

        # Transponer k
        proj_k = proj_k.transpose(0, 1)  # [batch_size, hidden_dim]

        # Calcular la matriz de atención
        att_scores = torch.matmul(proj_q, proj_k) / math.sqrt(proj_q.size(-1))  # [batch_size, hidden_dim]
        att_map = self.softmax(att_scores)  # [batch_size, hidden_dim]

        # Ponderar v con la matriz de atención
        att_output = torch.matmul(att_map, proj_v)  # [batch_size, hidden_dim]

        return att_output

class RFBMultiHAttnNetwork_V4(nn.Module):
    def __init__(self, query_dim, context_dim, filters_head):
        super(RFBMultiHAttnNetwork_V4, self).__init__()
        
        self.cross_attention = CrossAttentionEmbedding(query_dim=query_dim, context_dim=context_dim, 
                                                       filters_head=filters_head)
        self.batch_norm = nn.BatchNorm2d(filters_head)  
        self.dropout = nn.Dropout(0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        
        self.fc1 = nn.Linear(filters_head, 2) 

    def forward(self, query_input, context_input):
        
        attention_output = self.cross_attention(query_input, context_input)
        
        x = self.batch_norm(attention_output)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) 

        x = self.fc1(x)
        return x
 
class Embedding_RFBMultiHAttnNetwork_V4(nn.Module):
    def __init__(self, query_dim, context_dim, filters_head):
        super(Embedding_RFBMultiHAttnNetwork_V4, self).__init__()
        
        self.cross_attention = CrossAttentionEmbedding(query_dim=query_dim, context_dim=context_dim, 
                                                       filters_head=filters_head)
        self.batch_norm = nn.BatchNorm1d(filters_head)  # Cambiar a BatchNorm2d
        self.dropout = nn.Dropout(0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1) # Cambiar a AdaptiveAvgPool2d
        
        self.fc1 = nn.Linear(filters_head, 2) 

    def forward(self, query_input, context_input):
        att_output = self.cross_attention(query_input, context_input)
        att_output = att_output.unsqueeze(2)  # Añadir una dimensión para que sea compatible con AdaptiveAvgPool1d
        att_output = self.adaptive_pool(att_output)
        att_output = att_output.squeeze(2)  # Eliminar la dimensión adicional después del pooling
        att_output = self.batch_norm(att_output)
        att_output = self.dropout(att_output)
        att_output = self.fc1(att_output)
        return att_output

class BasicConv2D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), padding=1):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        #Agregar acá un batch norm
        return x
    

class New_RFBMultiHAttnNetwork_V4(nn.Module):
    def __init__(self, query_dim, context_dim, filters_head):
        super(New_RFBMultiHAttnNetwork_V4, self).__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.filters_head = filters_head

        # Define layers for attention mechanism
        self.query_conv = nn.Conv2d(query_dim, filters_head, kernel_size=1)
        self.key_conv = nn.Conv2d(context_dim, filters_head, kernel_size=1)
        self.value_conv = nn.Conv2d(context_dim, filters_head, kernel_size=1)
        self.out_conv = nn.Conv2d(filters_head, query_dim, kernel_size=1)

    def forward(self, query, context):
        # Adjust dimensions if needed
        if query.dim() == 2:  # (batch_size, embedding_dim)
            query = query.unsqueeze(-1)  # (batch_size, embedding_dim, 1)
        if context.dim() == 2:  # (batch_size, embedding_dim)
            context = context.unsqueeze(-1)  # (batch_size, embedding_dim, 1)

        # Apply convolutions
        query = self.query_conv(query)
        key = self.key_conv(context)
        value = self.value_conv(context)
        
        # Flatten and transpose for matrix multiplication
        query = query.view(query.size(0), self.filters_head, -1)
        key = key.view(key.size(0), self.filters_head, -1).transpose(1, 2)
        value = value.view(value.size(0), self.filters_head, -1)

        # Attention mechanism
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(attention, value)
        
        # Reshape and apply output convolution
        out = out.view(out.size(0), self.filters_head, -1)
        out = self.out_conv(out)

        return out    
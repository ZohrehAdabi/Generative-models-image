import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import itertools
import torch.nn.functional as F
from pytorch_revgrad import RevGrad

"""
Diffusion
"""
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=1000, original=False, device='cuda'):
        super().__init__()
        
        self.max_length = max_length
        self.embedding_dim = embedding_dim 
        self._max_period = 10_000.0
        self.original = original

        if self.original:
            self.positional_encodings = self.positional_encoding_original().to(device)
        else:
            self.positional_encodings = self.positional_encoding().to(device)

    def forward(self, time):
        
        return   self.positional_encodings[time.squeeze().long(), :]

    
    def positional_encoding(self):

        """
        From Fairseq.
        fairseq/fairseq/modules/sinusoidal_positional_embedding.py
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """   
        emb_dim = self.embedding_dim // 2
        pe = torch.zeros(self.max_length, emb_dim)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)

        scale = np.log(self._max_period) / (emb_dim-1)
        div_term = torch.exp(-scale * torch.arange(emb_dim))
        t_div = position * div_term[None, :]
        pe = torch.cat((t_div.sin(), t_div.cos()), dim=1)

        return pe

    def positional_encoding_original(self):
        """
        Sinusoidal Positional Encoding introduced by Vaswani et al. [1].
        Use a fixed trigonometric encoding for the position of an element in a sequence,

        [1] Vaswani et al. (https://arxiv.org/abs/1706.03762)

        Implementation inspired by `nncore`,
        link: https://github.com/yeliudev/nncore/blob/main/nncore/nn/blocks/transformer.py

        :param size: Size of each positional encoding vector.
        :param maximum_length: The maximum length of the input sequence.
        """

        pe = torch.zeros(self.max_length, self.embedding_dim)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)

        scale = np.log(self._max_period) / self.embedding_dim
        div_term = torch.exp(-scale * torch.arange(0, self.embedding_dim, 2).float())
        t_div = position * div_term
        pe[:, 0::2] = torch.sin(t_div)
        pe[:, 1::2] = torch.cos(t_div)

        return pe
    
    def get_positional_encoding(self, t):

        emb_dim = self.embedding_dim // 2 
        scale = np.log(self._max_period) / (emb_dim-1)
        div_term = torch.exp(-scale * torch.arange(emb_dim))
        t_div = t[:, None] * div_term[None, :]
        pe = torch.cat((t_div.sin(), t_div.cos()), dim=1)

        return pe

    def get_positional_encoding_original(self, t):

        pe = torch.zeros(t.shape[0], self.embedding_dim)

        scale = np.log(self._max_period) / self.embedding_dim
        div_term = torch.exp(-scale * torch.arange(0, self.embedding_dim, 2).float())
        t_div = t[:, None] * div_term[None, :]
        pe[:, 0::2] = torch.sin(t_div)
        pe[:, 1::2] = torch.cos(t_div)

        return pe
  

def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'swish':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')


class DDPM(nn.Module): 
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a time (without time embedding).
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(DDPM, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEncoding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        x_ = self.net1(torch.hstack([x, time])) # Add t to inputs
        # for net in self.net2:
        #     x_ = net(x_)
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out
   
class DDPMHidden(nn.Module):   
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a time (without time embedding).
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(DDPMHidden, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEncoding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        x_ = self.net1(torch.hstack([x, time])) # Add t to inputs
        # for net in self.net2:
        #     x_ = net(x_)
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out

class ScoreNet(nn.Module):   
    """
        For predicting score of distribution (with low level sigma)
    """
    def __init__(self, data_dim=2, hidden_dim=128, num_hidden=4, device='cuda'):
        super(ScoreNet, self).__init__()
        self.data_dim = data_dim
        

        # Make the network
        self.net1 = nn.Linear(data_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)


    def forward(self, x):
        """
            Forward pass of the network
        """
        x_ = self.net1(x) 
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out

class ScoreNetWithTime(nn.Module):   
    """
        For predicting score of distribution (with multi level sigma)
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(ScoreNetWithTime, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEncoding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        
        x_ = self.net1(torch.hstack([x, time])) # Add t to inputs
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out
  
class Boosting(nn.Module):  
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a time (without time embedding).
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(Boosting, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEncoding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        x_ = self.net1(torch.hstack([x, time.long()])) # Add t to inputs
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out

class BoostingTimeEmb(nn.Module):   
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a time (with time embedding).
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(BoostingTimeEmb, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEncoding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        time_embedding = self.positional_embedding(time.long())
        x_ = self.net1(torch.hstack([x, time_embedding])) # Add t to inputs
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out

class FlowMatching(nn.Module):  
    """
        Has a simple feed forward MLP structure. 

        Takes as input the data point and a time (without time embedding).
    """

    def __init__(self, data_dim=2, time_dim=2, hidden_dim=128, num_hidden=4, total_timesteps=1000, device='cuda'):
        super(FlowMatching, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        # Make the positional embedding
        self.positional_embedding = SinusoidalPositionalEncoding(
            time_dim, 
            total_timesteps,
            device=device
        )
        # Make the network
        self.net1 = nn.Linear(data_dim + time_dim, hidden_dim)
        # self.net2 = nn.ModuleList([DiffusionBlock(hidden_dim) for _ in range(4)])
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.net3 = nn.Linear(hidden_dim, data_dim)

        self.time_emb_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, time):
        """
            Forward pass of the network
        """
        x_ = self.net1(torch.hstack([x, time])) # Add t to inputs
        # for net in self.net2:
        #     x_ = net(x_)
        x_ = self.net2(x_)
        out = self.net3(x_)
        return out

class BoostingOne(nn.Module):          #prediction goal: noise of ddpm or grad of flow matching 
    """
    Individual network for each time t = > no time embedding
    """
    def __init__(self, data_dim=2, time_dim=None, hidden_dim=128, num_hidden=4, total_timesteps=40, device='cuda'):
        super(BoostingOne, self).__init__()

        self.total_timesteps = total_timesteps
        self.net = nn.Sequential(nn.Linear(data_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, hidden_dim),
                                                        nn.ReLU(), 
                                                        nn.Linear(hidden_dim, hidden_dim), 
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, data_dim)
                                                    )
        
       
    def predict(self, x, time, gamma):
        # h = self.net1(torch.concat([z, x], dim=-1))

        pred = x 
        for i in range(time, -1, -1):
            grad = self.net(pred)
            pred += gamma * grad
        return pred
    
    def predict_flow(self, x, time, gamma):
        # h = self.net1(torch.concat([z, x], dim=-1))

        pred = x 
        for i in range(time, -1, -1):
            grad = self.net(pred)
            pred += gamma * grad
        return pred
    
    def forward(self, x):

        return self.net(x)





class BasicUNetMNIST(nn.Module):    
                         # 02_diffusion_models_from_scratch
    """A minimal UNet implementation."""
    def __init__(self, in_channels=1, out_channels=1, base_channel=32, depth=2, time_emb_dim=1, device='cuda'):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, base_channel, kernel_size=5, padding=2),
            nn.Conv2d(base_channel, base_channel*2, kernel_size=5, padding=2),
            nn.Conv2d(base_channel*2, base_channel*2, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(base_channel*2, base_channel*2, kernel_size=5, padding=2),
            nn.Conv2d(base_channel*2, base_channel, kernel_size=5, padding=2),
            nn.Conv2d(base_channel, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # Through the layer and the activation function
            if i < 2: # For all but the third (final) down layer:
              h.append(x) # Storing output for skip connection
              x = self.downscale(x) # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0: # For all except the first up layer
              x = self.upscale(x) # Upscale
              x += h.pop() # Fetching stored output (skip connection)
            x = self.act(l(x)) # Through the layer and the activation function

        return x  





def conv_block(in_channels, out_channels, num_groups=8, dropout=False, ks=7, pd=3, cbt='simple'):    
    
    if cbt=='simple':
        """Basic convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=pd),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=ks, padding=pd),
            nn.ReLU(inplace=True)
        )
    if cbt=='with_batch_norm':
        """BatchNorm convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=pd),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=ks, padding=pd),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    elif cbt=='with_group_norm':
        """Advanced convolutional block with optional dropout."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=pd),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
            nn.Dropout(0.1 if dropout else 0),  # Add dropout with a probability of 0.1
            nn.Conv2d(out_channels, out_channels, kernel_size=ks, padding=pd),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU()
        )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8, dropout=False, ks=7, pd=3, cbt='simple'):
        super().__init__()
        self.conv_block = conv_block(in_channels, out_channels, num_groups, dropout, ks, pd, cbt)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = self.residual(x)  # residual connection
        x = self.conv_block(x)
        return x + residual

class BoostingOneUNetMNIST(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channel=16, depth=2, time_emb_dim=16, 
                 down=True, up=True, ks=7, pd=3, conv_block_type='simple', final_act=False,device='cuda'):
        super().__init__()
        
        # Time Embedding
        # self.time_embedding = nn.Embedding(1000, time_emb_dim)
        if time_emb_dim == 1:
            self.time_pos_enc = nn.Identity()
        else:
            self.time_pos_enc = SinusoidalPositionalEncoding(time_emb_dim)
        # layers_num = np.arange(0, depth)
        # hidden_channels =  [in_channels] + [base_channel*2**i for i in layers_num]
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
       
        for i in range(depth):
            self.encoder.append(conv_block(in_channels, base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))
            if down: self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks+1, stride=2, padding=pd))
            else:  # donot change size of image
                self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks, stride=1, padding=pd))
            in_channels = base_channel * 2 ** i

            # self.downsample.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(depth - 1, -1, -1):

            # self.upsample.append(nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False))
            if up:
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks+1, stride=2, padding=pd))
            else: # donot change size of image
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks, stride=1, padding=pd))
            self.decoder.append(conv_block(base_channel * 2 ** (i + 1), base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))


        if final_act:
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd), nn.Tanh())
        else: 
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))


    def forward(self, x):

        # t_emb = self.time_pos_enc(t) 
        #                     # [B, time_emb_dim, 1, 1]  # Expand to match spatial dimensions
        # t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # # Add time embedding to input
        # x = torch.cat((x, t_emb), dim=1)

        encoder_outputs = []
        for enc, down in zip(self.encoder, self.downsample):
            x = enc(x)
            encoder_outputs.append(x)
            x = down(x)
            
        x = self.bottleneck(x)

        for up, dec in zip(self.upsample, self.decoder):

            skip_connection = encoder_outputs.pop()
            x = up(x)
            x = torch.cat((x, skip_connection), dim=1)
            x = dec(x)          

        return self.out(x)
 
class DAEMNIST(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channel=16, depth=2, time_emb_dim=16, 
                 down=True, up=True, ks=7, pd=3, conv_block_type='simple', final_act=False,device='cuda'):
        super().__init__()
        
        # Time Embedding
        # self.time_embedding = nn.Embedding(1000, time_emb_dim)
        if time_emb_dim == 1:
            self.time_pos_enc = nn.Identity()
        else:
            self.time_pos_enc = SinusoidalPositionalEncoding(time_emb_dim)
        # layers_num = np.arange(0, depth)
        # hidden_channels =  [in_channels] + [base_channel*2**i for i in layers_num]
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
       
        for i in range(depth):
            self.encoder.append(conv_block(in_channels, base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))
            if down: self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks+1, stride=2, padding=pd))
            else:  # donot change size of image
                self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks, stride=1, padding=pd))
            in_channels = base_channel * 2 ** i

            # self.downsample.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(depth - 1, -1, -1):

            # self.upsample.append(nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False))
            if up:
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks+1, stride=2, padding=pd))
            else: # donot change size of image
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks, stride=1, padding=pd))
            self.decoder.append(conv_block(base_channel * 2 ** (i + 1), base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))


        if final_act:
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd), nn.Tanh())
        else: 
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))


    def forward(self, x):

        # t_emb = self.time_pos_enc(t) 
        #                     # [B, time_emb_dim, 1, 1]  # Expand to match spatial dimensions
        # t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # # Add time embedding to input
        # x = torch.cat((x, t_emb), dim=1)

        encoder_outputs = []
        for enc, down in zip(self.encoder, self.downsample):
            x = enc(x)
            encoder_outputs.append(x)
            x = down(x)
            
        x = self.bottleneck(x)

        for up, dec in zip(self.upsample, self.decoder):

            skip_connection = encoder_outputs.pop()
            x = up(x)
            x = torch.cat((x, skip_connection), dim=1)
            x = dec(x)          

        return self.out(x)
 
class VAEMNISTED(nn.Module): # encoder decoder, with out skip connection
    def __init__(self, in_channels=1, out_channels=1, base_channel=16, depth=2, time_emb_dim=16, 
                 down=True, up=True, ks=7, pd=3, conv_block_type='simple', final_act=False, device='cuda'):
        super().__init__()
        
        # Time Embedding
        # self.time_embedding = nn.Embedding(1000, time_emb_dim)
        if time_emb_dim == 1:
            self.time_pos_enc = nn.Identity()
        else:
            self.time_pos_enc = SinusoidalPositionalEncoding(time_emb_dim)
        # layers_num = np.arange(0, depth)
        # hidden_channels =  [in_channels] + [base_channel*2**i for i in layers_num]
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        # in_channels += time_emb_dim
        for i in range(depth):
            self.encoder.append(conv_block(in_channels, base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))
            if down: self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks+1, stride=2, padding=pd))
            else:  # donot change size of image
                self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks, stride=1, padding=pd))
            in_channels = base_channel * 2 ** i

            # self.downsample.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck_mean_z = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        self.bottleneck_var_z = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(depth - 1, -1, -1):

            # self.upsample.append(nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False))
            if up:
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** (i+1), 
                                                    kernel_size=ks+1, stride=2, padding=pd))
            else: # donot change size of image
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** (i+1), 
                                                    kernel_size=ks, stride=1, padding=pd))
            self.decoder.append(conv_block(base_channel * 2 ** (i + 1), base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))


        if final_act:
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd), nn.Tanh())
        else: 
            self.out_mean = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))
            self.out_var = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))


    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, t=None):
        
        
        # t_emb = self.time_pos_enc(t) 
        #                     # [B, time_emb_dim, 1, 1]  # Expand to match spatial dimensions
        # t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # # Add time embedding to input
        # x = torch.cat((x, t_emb), dim=1)

        encoder_outputs = []
        for enc, down in zip(self.encoder, self.downsample):
            x = enc(x)
            encoder_outputs.append(x)
            x = down(x)
            
        mu = self.bottleneck_mean_z(x)
        logvar = self.bottleneck_var_z(x)

        x = self.reparameterize(mu, logvar)

        for up, dec in zip(self.upsample, self.decoder):

            skip_connection = encoder_outputs.pop()
            x = up(x)
            # x = torch.cat((x, skip_connection), dim=1)
            x = dec(x)   

        out_mu = self.out_mean(x)
        out_logvar = self.out_var(x)

        return out_mu, out_logvar, mu, logvar
    
class EncoderMNIST(nn.Module): # # AE for encoding h|x,h, with out skip connection
    def __init__(self, in_channels=1, out_channels=1, base_channel=16, depth=2, time_emb_dim=16, 
                 down=True, up=True, ks=7, pd=3, conv_block_type='simple', final_act=False, device='cuda'):
        super().__init__()
        
        # Time Embedding
        # self.time_embedding = nn.Embedding(1000, time_emb_dim)
        if time_emb_dim == 1:
            self.time_pos_enc = nn.Identity()
        else:
            self.time_pos_enc = SinusoidalPositionalEncoding(time_emb_dim)
        # layers_num = np.arange(0, depth)
        # hidden_channels =  [in_channels] + [base_channel*2**i for i in layers_num]
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        # in_channels += time_emb_dim
        for i in range(depth):
            self.encoder.append(conv_block(in_channels, base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))
            if down: self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks+1, stride=2, padding=pd))
            else:  # donot change size of image
                self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks, stride=1, padding=pd))
            in_channels = base_channel * 2 ** i

            # self.downsample.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck_mean_z = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        self.bottleneck_var_z = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(depth - 1, -1, -1):

            # self.upsample.append(nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False))
            if up:
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** (i+1), 
                                                    kernel_size=ks+1, stride=2, padding=pd))
            else: # donot change size of image
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** (i+1), 
                                                    kernel_size=ks, stride=1, padding=pd))
            self.decoder.append(conv_block(base_channel * 2 ** (i + 1), base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))


        if final_act:
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd), nn.Tanh())
        else: 
            self.out_mean = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))
            self.out_var = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))


    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, xh, t=None):
        
        # t_emb = self.time_pos_enc(t) 
        #                     # [B, time_emb_dim, 1, 1]  # Expand to match spatial dimensions
        # t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # # Add time embedding to input
        # x = torch.cat((x, t_emb), dim=1)
        

        x = xh
        encoder_outputs = []
        for enc, down in zip(self.encoder, self.downsample):
            x = enc(x)
            # encoder_outputs.append(x)
            x = down(x)
               
        x = self.bottleneck_mean_z(x)
        # logvar = self.bottleneck_var_z(x)

        # x = self.reparameterize(mu, logvar)
        # z = x.clone()
        for up, dec in zip(self.upsample, self.decoder):

            # skip_connection = encoder_outputs.pop()
            x = up(x)
            # x = torch.cat((x, skip_connection), dim=1)
            x = dec(x)   

        out_mu = self.out_mean(x)
        out_logvar = self.out_var(x)

        return out_mu, out_logvar
    
class DecoderMNIST(nn.Module): # AE for decoding x|h, with out skip connection
    def __init__(self, in_channels=1, out_channels=1, base_channel=16, depth=2, time_emb_dim=16, 
                 down=True, up=True, ks=7, pd=3, conv_block_type='simple', final_act=False, device='cuda'):
        super().__init__()
        
        # Time Embedding
        # self.time_embedding = nn.Embedding(1000, time_emb_dim)
        if time_emb_dim == 1:
            self.time_pos_enc = nn.Identity()
        else:
            self.time_pos_enc = SinusoidalPositionalEncoding(time_emb_dim)
        # layers_num = np.arange(0, depth)
        # hidden_channels =  [in_channels] + [base_channel*2**i for i in layers_num]
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        # in_channels += time_emb_dim
        for i in range(depth):
            self.encoder.append(conv_block(in_channels, base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))
            if down: self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks+1, stride=2, padding=pd))
            else:  # donot change size of image
                self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks, stride=1, padding=pd))
            in_channels = base_channel * 2 ** i

            # self.downsample.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck_mean_z = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        self.bottleneck_var_z = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(depth - 1, -1, -1):

            # self.upsample.append(nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False))
            if up:
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** (i+1), 
                                                    kernel_size=ks+1, stride=2, padding=pd))
            else: # donot change size of image
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** (i+1), 
                                                    kernel_size=ks, stride=1, padding=pd))
            self.decoder.append(conv_block(base_channel * 2 ** (i + 1), base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))


        if final_act:
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd), nn.Tanh())
        else: 
            self.out_mean = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))
            self.out_var = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))


    def reparameterize(self, mu, logvar):
        # torch.isnan(std).any() or torch.isinf(std).any() or torch.isnan(mu).any() or torch.isinf(mu).any()
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, h, t=None):
        
        
        encoder_outputs = []
        for enc, down in zip(self.encoder, self.downsample):
            h = enc(h)
           # encoder_outputs.append(x)
            h = down(h)
            
        h = self.bottleneck_mean_z(h)
        # logvar = self.bottleneck_var_z(h)

        # h = self.reparameterize(mu, logvar)

        for up, dec in zip(self.upsample, self.decoder):

            # skip_connection = encoder_outputs.pop()
            h = up(h)
            # x = torch.cat((x, skip_connection), dim=1)
            h = dec(h)   

        out_mu = self.out_mean(h)
        out_logvar = self.out_var(h)

        return out_mu, out_logvar
    
class GSNUNetMNIST(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channel=16, depth=2, time_emb_dim=16, 
                 down=True, up=True, ks=7, pd=3, conv_block_type='simple', final_act=False, device='cuda'):
        super().__init__()
        
        # Time Embedding
        # self.time_embedding = nn.Embedding(1000, time_emb_dim)
        if time_emb_dim == 1:
            self.time_pos_enc = nn.Identity()
        else:
            self.time_pos_enc = SinusoidalPositionalEncoding(time_emb_dim)
        # layers_num = np.arange(0, depth)
        # hidden_channels =  [in_channels] + [base_channel*2**i for i in layers_num]
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        # in_channels += time_emb_dim
        for i in range(depth):
            self.encoder.append(conv_block(in_channels, base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))
            if down: self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks+1, stride=2, padding=pd))
            else:  # donot change size of image
                self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks, stride=1, padding=pd))
            in_channels = base_channel * 2 ** i

            # self.downsample.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(depth - 1, -1, -1):

            # self.upsample.append(nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False))
            if up:
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks+1, stride=2, padding=pd))
            else: # donot change size of image
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks, stride=1, padding=pd))
            self.decoder.append(conv_block(base_channel * 2 ** (i + 1), base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))


        if final_act:
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd), nn.Tanh())
        else: 
            self.out_mean = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))
            self.out_var = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))

        

    def forward(self, x, t=None):

        # t_emb = self.time_pos_enc(t) 
        #                     # [B, time_emb_dim, 1, 1]  # Expand to match spatial dimensions
        # t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # # Add time embedding to input
        # x = torch.cat((x, t_emb), dim=1)

        encoder_outputs = []
        for enc, down in zip(self.encoder, self.downsample):
            x = enc(x)
            encoder_outputs.append(x)
            x = down(x)
            
        x = self.bottleneck(x)

        for up, dec in zip(self.upsample, self.decoder):

            skip_connection = encoder_outputs.pop()
            x = up(x)
            x = torch.cat((x, skip_connection), dim=1)
            x = dec(x)   

        out_mu = self.out_mean(x)
        out_logvar = self.out_var(x)

        return out_mu, out_logvar


class VAEHiddenMNISTED(nn.Module): # encoder decoder, with out skip connection
    def __init__(self, in_channels=1, out_channels=1, base_channel=16, depth=2, time_emb_dim=16, 
                 down=True, up=True, ks=7, pd=3, conv_block_type='simple', final_act=False, device='cuda'):
        super().__init__()
        
        # Time Embedding

        #*********************************************************************************
        # AE for encoding h|x,h
        self.encoder = EncoderMNIST(2, out_channels, base_channel, depth, time_emb_dim, 
                 down, up, ks, pd, conv_block_type, final_act, device)
        #***************************************************************************
        # AE for decoding x|h
        self.decoder = DecoderMNIST(in_channels, out_channels, base_channel, depth, time_emb_dim, 
                 down, up, ks, pd, conv_block_type, final_act, device)


    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, h, burn_in=False, n_burn_in=0, fixed_encoder=False, t=None):
        
        if fixed_encoder:
            if burn_in:
                x0 = x.clone()
                for i in range(n_burn_in):
                    
                    x = x0 + h 
                    eps = torch.randn_like(x)
                    h = x + eps
                h = x0 + h

            else:
                eps = torch.randn_like(x)
                h = x + h + eps
                
        else:
            if burn_in:
                x0 = x.clone()
                for i in range(n_burn_in):
                    
                    # x = x0 + h
                    x = torch.cat((x0, h), dim=1)
                    h_mu, h_logvar = self.encoder(x)
                    h = self.reparameterize(h_mu, h_logvar)
                # x = x0 + h
                x = torch.cat((x0, h), dim=1)

            else:
                # x = x + h
                x = torch.cat((x, h), dim=1)

            h_mu, h_logvar = self.encoder(x)
            h = self.reparameterize(h_mu, h_logvar)

        out_mu, out_logvar = self.decoder(h)

        return out_mu, out_logvar, h
   
class VAEHiddenUnetMNISTED(nn.Module): # encoder decoder, with out skip connection
    def __init__(self, in_channels=1, out_channels=1, base_channel=16, depth=2, time_emb_dim=16, 
                 down=True, up=True, ks=7, pd=3, conv_block_type='simple', final_act=False, device='cuda'):
        super().__init__()
        
        # Time Embedding

        #*********************************************************************************
        # AE for encoding h|x,h
        self.encoder = GSNUNetMNIST(2, out_channels, base_channel, depth, time_emb_dim, 
                 down, up, ks, pd, conv_block_type, final_act, device)
        #***************************************************************************
        # AE for decoding x|h
        self.decoder = GSNUNetMNIST(in_channels, out_channels, base_channel, depth, time_emb_dim, 
                 down, up, ks, pd, conv_block_type, final_act, device)


    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, h, burn_in=False, n_burn_in=0, fixed_encoder=False, t=None):
        
        if fixed_encoder:
            if burn_in:
                x0 = x.clone()
                for i in range(n_burn_in):
                    
                    x = x0 + h 
                    eps = torch.randn_like(x)
                    h = x + eps
                h = x0 + h

            else:
                eps = torch.randn_like(x)
                h = x + h + eps
                
        else:
            if burn_in:
                x0 = x.clone()
                for i in range(n_burn_in):
                    
                    # x = x0 + h
                    x = torch.cat((x0, h), dim=1)
                    h_mu, h_logvar = self.encoder(x)
                    h = self.reparameterize(h_mu, h_logvar)
                # x = x0 + h
                x = torch.cat((x0, h), dim=1)

            else:
                # x = x + h
                x = torch.cat((x, h), dim=1)

            h_mu, h_logvar = self.encoder(x)
            h = self.reparameterize(h_mu, h_logvar)

        out_mu, out_logvar = self.decoder(h)

        return out_mu, out_logvar, h.detach()
 

class VAEMNIST(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channel=16, depth=2, time_emb_dim=16, 
                 down=True, up=True, ks=7, pd=3, conv_block_type='simple', final_act=False, device='cuda'):
        super().__init__()
        
        # Time Embedding
        # self.time_embedding = nn.Embedding(1000, time_emb_dim)
        if time_emb_dim == 1:
            self.time_pos_enc = nn.Identity()
        else:
            self.time_pos_enc = SinusoidalPositionalEncoding(time_emb_dim)
        # layers_num = np.arange(0, depth)
        # hidden_channels =  [in_channels] + [base_channel*2**i for i in layers_num]
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        # in_channels += time_emb_dim
        for i in range(depth):
            self.encoder.append(conv_block(in_channels, base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))
            if down: self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks+1, stride=2, padding=pd))
            else:  # donot change size of image
                self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks, stride=1, padding=pd))
            in_channels = base_channel * 2 ** i

            # self.downsample.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck_mean_z = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        self.bottleneck_var_z = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(depth - 1, -1, -1):

            # self.upsample.append(nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False))
            if up:
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks+1, stride=2, padding=pd))
            else: # donot change size of image
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks, stride=1, padding=pd))
            self.decoder.append(conv_block(base_channel * 2 ** (i + 1), base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))


        if final_act:
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd), nn.Tanh())
        else: 
            self.out_mean = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))
            self.out_var = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))


    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, t=None):
        
        
        # t_emb = self.time_pos_enc(t) 
        #                     # [B, time_emb_dim, 1, 1]  # Expand to match spatial dimensions
        # t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # # Add time embedding to input
        # x = torch.cat((x, t_emb), dim=1)

        encoder_outputs = []
        for enc, down in zip(self.encoder, self.downsample):
            x = enc(x)
            encoder_outputs.append(x)
            x = down(x)
            
        mu = self.bottleneck_mean_z(x)
        logvar = self.bottleneck_var_z(x)

        x = self.reparameterize(mu, logvar)

        for up, dec in zip(self.upsample, self.decoder):

            skip_connection = encoder_outputs.pop()
            x = up(x)
            x = torch.cat((x, skip_connection), dim=1)
            x = dec(x)   

        out_mu = self.out_mean(x)
        out_logvar = self.out_var(x)

        return out_mu, out_logvar, mu, logvar
    

class UNetMNIST(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channel=16, depth=2, time_emb_dim=16, 
                 down=True, up=True, ks=7, pd=3, conv_block_type='simple', final_act=False, device='cuda'):
        super().__init__()
        
        # Time Embedding
        # self.time_embedding = nn.Embedding(1000, time_emb_dim)
        if time_emb_dim == 1:
            self.time_pos_enc = nn.Identity()
        else:
            self.time_pos_enc = SinusoidalPositionalEncoding(time_emb_dim)
        # layers_num = np.arange(0, depth)
        # hidden_channels =  [in_channels] + [base_channel*2**i for i in layers_num]
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_channels += time_emb_dim
        for i in range(depth):
            self.encoder.append(conv_block(in_channels, base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))
            if down: self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks+1, stride=2, padding=pd))
            else:  # donot change size of image
                self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks, stride=1, padding=pd))
            in_channels = base_channel * 2 ** i

            # self.downsample.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(depth - 1, -1, -1):

            # self.upsample.append(nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False))
            if up:
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks+1, stride=2, padding=pd))
            else: # donot change size of image
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks, stride=1, padding=pd))
            self.decoder.append(conv_block(base_channel * 2 ** (i + 1), base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))


        if final_act:
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd), nn.Tanh())
        else: 
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))

        self.reverse_grad = RevGrad()

    def forward(self, x, t, rvrs_grad=False):

        t_emb = self.time_pos_enc(t) 
                            # [B, time_emb_dim, 1, 1]  # Expand to match spatial dimensions
        t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # Add time embedding to input
        x = torch.cat((x, t_emb), dim=1)

        encoder_outputs = []
        for enc, down in zip(self.encoder, self.downsample):
            x = enc(x)
            encoder_outputs.append(x)
            x = down(x)
            
        x = self.bottleneck(x)

        for up, dec in zip(self.upsample, self.decoder):

            skip_connection = encoder_outputs.pop()
            x = up(x)
            x = torch.cat((x, skip_connection), dim=1)
            x = dec(x)   

        out = self.out(x)

        if rvrs_grad:
            return self.reverse_grad(out)
        else: return out


class ResUNetMNIST(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channel=16, depth=2, time_emb_dim=16, 
                 down=True, up=True, ks=7, pd=3, conv_block_type='simple', final_act=False, device='cuda'):
        super().__init__()
        
        # Time Embedding
        # self.time_embedding = nn.Embedding(1000, time_emb_dim)
        if time_emb_dim == 1:
            self.time_pos_enc = nn.Identity()
        else:
            self.time_pos_enc = SinusoidalPositionalEncoding(time_emb_dim)
        # layers_num = np.arange(0, depth)
        # hidden_channels =  [in_channels] + [base_channel*2**i for i in layers_num]
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_channels += time_emb_dim
        for i in range(depth):
            self.encoder.append(ResidualBlock(in_channels, base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))
            if down: self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks+1, stride=2, padding=pd))
            else:  # donot change size of image
                self.downsample.append(nn.Conv2d(base_channel * 2 ** i, base_channel * 2 ** i, 
                                             kernel_size=ks, stride=1, padding=pd))
            in_channels = base_channel * 2 ** i

            # self.downsample.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(base_channel*2**(depth-1), base_channel*2**(depth), 
                                                  kernel_size=ks, padding=pd), nn.LeakyReLU(inplace=True))
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(depth - 1, -1, -1):

            # self.upsample.append(nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False))
            if up:
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks+1, stride=2, padding=pd))
            else: # donot change size of image
                self.upsample.append(nn.ConvTranspose2d(base_channel * 2 ** (i + 1), base_channel * 2 ** i, 
                                                    kernel_size=ks, stride=1, padding=pd))
            self.decoder.append(ResidualBlock(base_channel * 2 ** (i + 1), base_channel * 2 ** i, ks=ks, pd=pd, cbt=conv_block_type))


        if final_act:
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd), nn.Tanh())
        else: 
            self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=ks, padding=pd))

        self.reverse_grad = RevGrad()

    def forward(self, x, t, rvrs_grad=False):

        t_emb = self.time_pos_enc(t) 
                            # [B, time_emb_dim, 1, 1]  # Expand to match spatial dimensions
        t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # Add time embedding to input
        x = torch.cat((x, t_emb), dim=1)

        encoder_outputs = []
        for enc, down in zip(self.encoder, self.downsample):
            x = enc(x)
            encoder_outputs.append(x)
            x = down(x)
            
        x = self.bottleneck(x)

        for up, dec in zip(self.upsample, self.decoder):

            skip_connection = encoder_outputs.pop()
            x = up(x)
            x = torch.cat((x, skip_connection), dim=1)
            x = dec(x)   

        out = self.out(x)

        if rvrs_grad:
            return self.reverse_grad(out)
        else: return out



blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channel=32, depth=2, time_emb_dim=16, device='cuda'):
        super(DummyEpsModel, self).__init__()
        if depth == 2:
            self.conv = nn.Sequential(  # with batchnorm
                blk(in_channels, base_channel),
                blk(base_channel, base_channel*2),
                blk(base_channel*2, base_channel*2),
                blk(base_channel*2, base_channel),
                nn.Conv2d(base_channel, out_channels, 3, padding=1),
            )
        elif depth == 3:
            self.conv = nn.Sequential(  # with batchnorm
                blk(in_channels, base_channel),
                blk(base_channel, base_channel*2),
                blk(base_channel*2, base_channel*4),
                blk(base_channel*4, base_channel*2),
                blk(base_channel*2, base_channel),
                nn.Conv2d(base_channel, out_channels, 3, padding=1),
            )
        elif depth == 4:
            self.conv = nn.Sequential(  # with batchnorm
                blk(in_channels, base_channel),
                blk(base_channel, base_channel*2),
                blk(base_channel*2, base_channel*4),
                blk(base_channel*4, base_channel*8),
                blk(base_channel*8, base_channel*4),
                blk(base_channel*4, base_channel*2),
                blk(base_channel*2, base_channel),
                nn.Conv2d(base_channel, out_channels, 3, padding=1),
            )

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)
    

class UNetCIFAR10(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channel=16, depth=4, time_emb_dim=16, device='cuda'):
        super().__init__()
        
        # self.time_embedding = nn.Embedding(1000, time_emb_dim)
        self.time_pos_enc = SinusoidalPositionalEncoding(time_emb_dim)
        layers_num = np.arange(0, depth)
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels + time_emb_dim, base_channel, kernel_size=3, padding=1), nn.ReLU(inplace=True))] +
            [nn.Sequential(nn.Conv2d(base_channel*2**i, base_channel*2**(i+1), kernel_size=3, padding=1), nn.ReLU(inplace=True)) for i in layers_num[:-1]])
        
        self.downsample = nn.MaxPool2d(2) 
        
        self.bottleneck = nn.Sequential(nn.Conv2d(base_channel*2**(layers_num[-1]), base_channel*2**(layers_num[-1]+1), kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.decoder = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(base_channel*2**(i+1), base_channel*2**i, kernel_size=3, padding=1), nn.ReLU(inplace=True)) for i in layers_num[::-1]])
        
        self.upsample = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(base_channel*2**(i+1), base_channel*2**i, kernel_size=2, stride=2), nn.ReLU(inplace=True)) for i in layers_num[::-1]])
        
        
        self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=1), nn.Tanh())

    def forward(self, x, t):
        t_emb = self.time_pos_enc(t)
        t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat((x, t_emb), dim=1)

        encoder_outputs = []
        for encode in self.encoder:
            x = encode(x)
            encoder_outputs.append(x)
            x = self.downsample(x)
            
        x = self.bottleneck(x)

        for up, decode in zip(self.upsample, self.decoder):
            skip_connection = encoder_outputs.pop()
            x = up(x)
            x = torch.cat((x, skip_connection), dim=1) # Concatenate skip connection
            x = decode(x)           

        return self.out(x)

"""
GAN
"""

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channel=16, data_dim=32, depth=2):
        super().__init__()
        

        layers_num = np.arange(0, depth)
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, base_channel, kernel_size=3, padding=1), nn.ReLU(inplace=True))] +
            [nn.Sequential(nn.Conv2d(base_channel*2**i, base_channel*2**(i+1), kernel_size=3, padding=1), nn.ReLU(inplace=True)) for i in layers_num[:-1]])
        
        self.downsample = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(base_channel*2**(layers_num[-1]), base_channel*2**(layers_num[-1]+1), kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.x_dim = data_dim // 2**(layers_num[-1]+1)
        self.out = nn.Sequential(nn.Linear(in_features=base_channel*2**(layers_num[-1]+1)*(self.x_dim**2), out_features=1), nn.Sigmoid())

    def forward(self, x):
        
        for encode in self.encoder:
            x = encode(x)
            x = self.downsample(x)
        x = self.bottleneck(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

class Generator(nn.Module):
    def __init__(self, in_channels=8, out_channels=1, base_channel=16, depth=2):
        super().__init__()
        

        layers_num = np.arange(0, depth)

        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels, base_channel*2**(layers_num[-1]+1), kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.decoder = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(base_channel*2**(i+1), base_channel*2**i, kernel_size=3, padding=1), nn.ReLU(inplace=True)) for i in layers_num[::-1]])
        
        self.upsample = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(base_channel*2**(i+1), base_channel*2**(i+1), kernel_size=2, stride=3), nn.ReLU(inplace=True)) for i in layers_num[::-1]])
        
        
        self.out = nn.Sequential(nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=1), nn.Tanh())

    def forward(self, x):

        x = x.view(x.size(0), -1, 1, 1).expand(x.size(0), -1, 4, 4) # Reshape latent vector    
        x = self.bottleneck(x)
        
        for up, decode in zip(self.upsample, self.decoder):
            
            x = up(x)
            # print(f'up {x.shape}')
            x = decode(x)  
            # print(f'decode {x.shape}')         

        return self.out(x)

class GAN(nn.Module):
    def __init__(self, z_dim=8, in_channels=8, out_channels=1, base_channel=16, data_dim=32, depth=2, device='cuda'):
        super().__init__()

        self.generator = Generator(in_channels=z_dim, out_channels=out_channels, base_channel=base_channel, depth=depth) #
        self.discriminator = Discriminator(in_channels=in_channels, base_channel=base_channel, data_dim=data_dim, depth=depth)



def select_model_diffusion(model_info, time_dim, n_timesteps, data_dim=None,device='cuda'):
    
    model_info = model_info.split('_')
    model_name = model_info[0]
    num_hidden, hidden_dim = 1, 1
    down, up = True, True
    ks, pd = 3, 1
    final_act = False
    num_params = len(model_info)
    if num_params > 1:
        num_hidden = int(model_info[1])
        hidden_dim = int(model_info[2])
        conv_block_type = 'simple'

        if 'BN' in model_info:
            conv_block_type =  'with_batch_norm' 
        elif 'GN' in model_info:
            conv_block_type =  'with_group_norm' 
        if 'ND' in model_info:
            down =  False
        if 'NU' in model_info:
            up =  False 
        if 'kp' in model_info: 
            ix = model_info.index('kp')
            ks, pd = int(model_info[ix+1][0]), int(model_info[ix+1][1])
        if 'fa' in model_info:
            final_act = True

    if model_name=='DDPM':
        return DDPM(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='DDPMHidden':
        return DDPM(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ScoreNet':
        return ScoreNet(data_dim=data_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, device=device)
    elif model_name=='ScoreNetWithTime':
        return ScoreNetWithTime(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='ToyBoosting':
        return Boosting(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='BoostingTimeEmb':
        return BoostingTimeEmb(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='FlowMatching':
        return FlowMatching(data_dim=data_dim, time_dim=time_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, total_timesteps=n_timesteps, device=device)
    elif model_name=='BoostingOneUNetMNIST':
        return BoostingOneUNetMNIST(in_channels=1, out_channels=1, base_channel=hidden_dim, depth=num_hidden, 
                                    down=down, up=up, ks=ks, pd=pd, conv_block_type=conv_block_type, final_act=final_act, device=device)
    elif model_name=='DAEMNIST':
        return DAEMNIST(in_channels=1, out_channels=1, base_channel=hidden_dim, depth=num_hidden, 
                                    down=down, up=up, ks=ks, pd=pd, conv_block_type=conv_block_type, final_act=final_act, device=device)
    elif model_name=='VAEMNIST':
        return VAEMNIST(in_channels=1, out_channels=1, base_channel=hidden_dim, depth=num_hidden, 
                                    down=down, up=up, ks=ks, pd=pd, conv_block_type=conv_block_type, final_act=final_act, device=device) 
    elif model_name=='VAEHiddenMNISTED':
        return VAEHiddenMNISTED(in_channels=1, out_channels=1, base_channel=hidden_dim, depth=num_hidden, 
                                    down=down, up=up, ks=ks, pd=pd, conv_block_type=conv_block_type, final_act=final_act, device=device) 
    elif model_name=='VAEHiddenUnetMNISTED':
        return VAEHiddenUnetMNISTED(in_channels=1, out_channels=1, base_channel=hidden_dim, depth=num_hidden, 
                                    down=down, up=up, ks=ks, pd=pd, conv_block_type=conv_block_type, final_act=final_act, device=device) 
    elif model_name=='VAEMNISTED':
        return VAEMNISTED(in_channels=1, out_channels=1, base_channel=hidden_dim, depth=num_hidden, 
                                    down=down, up=up, ks=ks, pd=pd, conv_block_type=conv_block_type, final_act=final_act, device=device)    
    elif model_name=='BasicUNetMNIST':
        return BasicUNetMNIST(in_channels=1, out_channels=1, base_channel=hidden_dim, depth=num_hidden, time_emb_dim=time_dim, device=device)
    elif model_name=='DummyEpsModel':
        return DummyEpsModel(in_channels=1, out_channels=1, base_channel=hidden_dim, depth=num_hidden, time_emb_dim=time_dim, device=device)
    elif model_name=='UNetMNIST':
        return UNetMNIST(in_channels=1, out_channels=1, base_channel=hidden_dim, depth=num_hidden, time_emb_dim=time_dim, 
                         down=down, up=up, ks=ks, pd=pd, conv_block_type=conv_block_type, final_act=final_act, device=device)
    elif model_name=='ResUNetMNIST':
        return ResUNetMNIST(in_channels=1, out_channels=1, base_channel=hidden_dim, depth=num_hidden, time_emb_dim=time_dim, 
                         down=down, up=up, ks=ks, pd=pd, conv_block_type=conv_block_type, final_act=final_act, device=device)
    elif model_name=='UNetCIFAR10':
        return UNetCIFAR10(in_channels=3, out_channels=3, base_channel=hidden_dim, depth=num_hidden, time_emb_dim=time_dim, device=device)

def select_model_gan(model_info, data_dim, z_dim, device='cuda'):
    model_info = model_info.split('_')
    model_name = model_info[0]
    num_hidden = int(model_info[1])
    hidden_dim = int(model_info[2])
    
    if model_name=='GANMNIST':
        return GAN(z_dim=z_dim, in_channels=1, out_channels=1, base_channel=hidden_dim, data_dim=data_dim, depth=num_hidden,device=device)
    if model_name=='GANCIFAR10':
        return GAN(z_dim=z_dim, in_channels=3, out_channels=3, base_channel=hidden_dim, data_dim=data_dim, depth=num_hidden,device=device)
    



if __name__=='__main__':

    # net1 = DDPM(time_dim=8)
    # net2 = DDPMHidden(time_dim=8)
    # net3 = FlowMatching(time_dim=8)
    # net4 = Boosting(time_dim=1)
    # net5 = BoostingOne(time_dim=1)
    # net6 = GAN(time_dim=1)

    # out1 = net1(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([10.0], dtype=torch.float))
    # out2 = net2(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([10.0], dtype=torch.float))
    # out3 = net3(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([10.0], dtype=torch.float))
    # out4 = net4(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([[10.0]], dtype=torch.float))
    # out5 = net5(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([10.0], dtype=torch.float))
    # out6 = net6(torch.tensor([[1.0, 2.0]], dtype=torch.float), torch.tensor([10.0], dtype=torch.float))

    # unet1 = BasicUNet()
    # out = unet1(torch.rand([28, 28], dtype=torch.float).view(1, 1, 28, 28))

    embedding_dim = 64
    time_emb_orig = SinusoidalPositionalEncoding(64, 100, True)
    time_emb = SinusoidalPositionalEncoding(64, 100)
    t = torch.tensor([5, 6]).to('cuda')
    te_org = time_emb_orig(t)
    te = time_emb(t)

    te_org.shape, te.shape

    # Test the model
    mnist_model = UNetMNIST(base_channel=16, depth=2).to('cuda')
    x_mnist = torch.randn(4, 1, 32, 32).to('cuda')  # Example MNIST batch
    # Should output a tensor with shape [4, 1, 28, 28]
    output_mnist = mnist_model(x_mnist, torch.tensor([2, 3, 4, 5], device='cuda').view(4, -1)) 

    print(f"MNIST output shape: {output_mnist.shape}")

    # Test the model
    mnist_model = BasicUNetMNIST().to('cuda')
    x_mnist = torch.randn(4, 1, 32, 32).to('cuda')  # Example MNIST batch
    # Should output a tensor with shape [4, 1, 28, 28]
    output_mnist = mnist_model(x_mnist) 

    print(f"MNIST output shape: {output_mnist.shape}")
    

    # Test the model
    cifar10_model = UNetCIFAR10().to('cuda')
    x_cifar10 = torch.randn(4, 3, 32, 32).to('cuda')  # Example MNIST batch
    # Should output a tensor with shape [4, 3, 28, 28]
    output_cifar10 = cifar10_model(x_cifar10, torch.tensor([2, 3, 4, 5], device='cuda').view(4, -1)) 

    print(f"CIFAR10 output shape: {output_cifar10.shape}")

    x_mnist = torch.randn(4, 1, 32, 32)
    x_cifar10 = torch.randn(4, 3, 32, 32)
    dsc_mnist = Discriminator(in_channels=1)
    dsc_cifar10 = Discriminator(in_channels=3)
    print(f"MNIST output {dsc_mnist(x_mnist)}")
    print(f"CIFAR10 output {dsc_cifar10(x_cifar10)}")


    z = torch.randn(4, 8)
    gen_mnist = Generator(in_channels=z.shape[1], out_channels=1)
    gen_cifar10 = Generator(in_channels=z.shape[1], out_channels=3)
    print(gen_mnist(z).shape)
    print(gen_cifar10(z).shape)
    print()
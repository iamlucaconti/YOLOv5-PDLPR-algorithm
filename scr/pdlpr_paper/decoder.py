import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor import CNNBlock
from attention import SelfAttention, CrossAttention
from encoder import PositionalEncoding

# output Encoder -> (B, 512, 6, 18) is the input of CNN BLOCK3

# NOTE: CNN BLOCK3 ha stride=3, kernelsize (2,1), padding=1, out_dim=512
# (B, 512, 6, 18) -> (B, 512, 3, 7) 

# NOTE: CNN BLOCK4 ha stride=3, kernelsize 1, padding=(0,1), out_dim=512
# (B, 512, 1, 3)

class DecoderUnit(nn.Module):
    def __init__(self, d_embed: int, d_cross: int, n_heads: int):
        super().__init__()


        self.mskd_attn = SelfAttention(n_heads=n_heads, d_embed=d_embed)
        self.cross_attn = CrossAttention(n_heads=n_heads, d_embed=d_embed, d_cross=d_cross)
        self.norm1 = nn.LayerNorm(d_embed)  # NOTE: e se facessimo Group Norm?
        self.norm2 = nn.LayerNorm(d_embed)  # NOTE: e se facessimo Group Norm?
        self.norm3 = nn.LayerNorm(d_embed)
        self.linear1 = nn.Linear(d_embed, d_embed)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_embed, d_embed)
        

    def forward(self, x, enc_out):
        # x: (B, in_channels, H, W)

        residual = x
        
        b, c, h, w = x.shape
        # (B, in_channels, H, W) -> (B, in_channels, H * W) -> (B, H * W, d_embed)
        x = x.view((b, c, h * w)).transpose(-1, -2)
        # Masked Self-attention
        # (B, H * W, in_channels) -> (B, H * W, d_embed)
        x = self.mskd_attn(x, causal_mask=True)
        # (B, H * W, d_embed) ->  (B, C, H * d_embed)  -> (B, d_embed, H , W) 
        x = x.transpose(-1, -2).view((b, c, h, w))
    
        # (B, d_embed, H , W)  -> (B, d_embed, H , W) 
        x = x + residual
        # (B, d_embed, H , W) -> # (B, H , W, d_embed)
        x = x.permute(0, 2, 3, 1)
        # (B, H , W, d_embed) -> (B, H , W, d_embed)
        x = self.norm1(x) 
        # (B, H , W, d_embed) -> (B, d_embed, H , W)
        x = x.permute(0, 3, 1, 2)
        residual = x

        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # (B, Dim_KV, Hy, Wy) -> (B, Seq_Len_KV, Dim_KV)
        y = enc_out.squeeze(2)
        x = x.view((b, c, h * w)).transpose(-1, -2)
        # (B, H * W, d_embed)
        x = self.cross_attn(x, y)
        # (B, H * W, d_embed) ->  (B, C, H * d_embed) -> (B, d_embed, H , W) 
        x = x.transpose(-1, -2).view((b, c, h, w))

        # (B, d_embed, H , W)  -> (B, d_embed, H , W) 
        x = x + residual
        # (B, d_embed, H , W) -> # (B, H , W, d_embed)
        x = x.permute(0, 2, 3, 1)
        # # (B, H , W, d_embed) -> (B, H , W, d_embed)
        x = self.norm2(x)
        # (B, H, W, d_embed) -> (B, d_embed, H , W)
        x = x.permute(0, 3, 1, 2)

        residual = x

        # (B, W, d_embed, H , W) -> (B, H*W, d_embed)
        x = x.reshape(b, h*w, c)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # (B, H*W, d_embed) -> (B, H, W, d_embed) -> (B, d_embed, H, W)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)

        # (B, d_embed, H , W)  -> (B, d_embed, H , W) 
        x = x + residual
        # (B, d_embed, H , W) -> # (B, H , W, d_embed)
        x = x.permute(0, 2, 3, 1)
        # # (B, H , W, d_embed) -> (B, H , W, d_embed)
        x = self.norm3(x)
        # (B, H , W, W, d_embed) -> (B, W, d_embed, H , W)
        x = x.permute(0, 3, 1, 2)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 height = 6,
                 width = 18,
                 d_embed=512,
                 d_cross=3,
                 dec_unit=3,
                 n_heads=8):
        super().__init__()

        self.height = height
        self.width = width
        self.seq_len = height * width
        self.d_model = d_embed

        self.pos_encoder = PositionalEncoding(d_model=self.d_model,
                                              seq_len=self.seq_len)
        self.layers = nn.ModuleList([
            DecoderUnit(d_embed=d_embed, d_cross=d_cross, n_heads=n_heads)
            for _ in range(dec_unit)
        ])

    def forward(self, x, conv_out):
        # x: (B, in_channels, H, W)
        # conv_out: (B, d_embed, a, b)
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x = x.flatten(2).permute(0, 2, 1)
        # (B, H*W, C)
        x = self.pos_encoder(x)
        # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        for layer in self.layers:
            x = layer(x, conv_out)
        return x

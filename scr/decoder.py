import torch
import torch.nn as nn
from attention import SelfAttention, CrossAttention
from encoder import PositionalEncoding, AddAndNorm


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.linear1 = nn.Linear(d_embed, d_embed)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_embed, d_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
         # x: (B, d_embed, H , W)
        b, c, h, w = x.shape
        # (B, d_embed, H , W) -> (B, H*W, d_embed)
        x = x.reshape(b, h*w, c)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # (B, H*W, d_embed) -> (B, H, W, d_embed) -> (B, d_embed, H, W)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        return x


class DecoderUnit(nn.Module):
    def __init__(self, d_embed: int, d_cross: int, n_heads: int):
        super().__init__()

        self.mskd_attn = SelfAttention(n_heads=n_heads, d_embed=d_embed)
        self.cross_attn = CrossAttention(n_heads=n_heads, d_embed=d_embed, d_cross=d_cross)
        self.addnorm1 = AddAndNorm(d_embed)
        self.addnorm2 = AddAndNorm(d_embed)
        self.addnorm3 = AddAndNorm(d_embed)
        self.ffn = FeedForwardNetwork(d_embed)
 

    def forward(self, x, y):
        # x: (B, in_channels, H, W)
        # y: (B, in_channels, hy, wy)

        
        # 1. Masked Self-attention
        residual = x
        # (B, in_channels, H, W) -> (B, d_embed, H , W) 
        x = self.mskd_attn(x, causal_mask=False)
        # (B, d_embed, H , W)  -> (B, d_embed, H , W) 
        x = self.addnorm1(x, residual)

        # 2. Cross attention
        residual = x
        # (B, d_embed, H , W)  -> (B, d_embed, H , W)
        x = self.cross_attn(x, y)
        # (B, d_embed, H , W)  -> (B, d_embed, H , W) 
        x = self.addnorm2(x, residual)

        #3. FFN
        residual = x
        # (B, W, d_embed, H , W) -> (B, W, d_embed, H , W)
        x = self.ffn(x)
        # (B, d_embed, H , W)  -> (B, d_embed, H , W) 
        x = self.addnorm3(x, residual) 

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
        self.d_model = d_embed

        self.pos_encoder = PositionalEncoding(d_model=self.d_model,
                                              height=self.height,
                                              width=self.width)
        
        self.layers = nn.ModuleList([
            DecoderUnit(d_embed=d_embed, d_cross=d_cross, n_heads=n_heads)
            for _ in range(dec_unit)
        ])

    def forward(self, x, conv_out):
        # x: (B, in_channels, H, W)
        # conv_out: (B, d_embed, a, b)

        # (B, in_channels, H, W) -> (B, in_channels, H, W)
        x = self.pos_encoder(x)
        
        # (B, in_channels, H, W) -> (B, in_channels, H, W)
        for layer in self.layers:
            x = layer(x, conv_out)
        return x # (B, in_channels, H, W)

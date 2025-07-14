import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor import CNNBlock
from attention import SelfAttention
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len) 
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 1D tensor -> (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return x


class EncoderUnit(nn.Module):
    def __init__(self, in_channels: int, d_embed: int, n_heads: int):
        super().__init__()
        # NOTE: [!] se metto padding = 1 le dimensioni di x aumentano di 2 sia di height che di width
        self.cnn1 = CNNBlock(in_channels=in_channels, out_channels=d_embed, kernel_size=1, stride=1, padding=0)
        self.attention = SelfAttention(n_heads=n_heads, d_embed=d_embed)
        # NOTE: [!] se metto padding = 1 le dimensioni di x aumentano
        self.cnn2 = CNNBlock(in_channels=d_embed, out_channels=in_channels, kernel_size=1, stride=1, padding=0) 
        self.norm = nn.LayerNorm(in_channels)  # NOTE: e se facessimo Group Norm?

    def forward(self, x):
        # x: (B, in_channels, H, W)
        residual = x
        # (B, in_channels, H, W) -> (B, d_embed, H, W)
        x = self.cnn1(x)
        b, c, h, w = x.shape

        # (B, d_embed, H, W) -> (B, d_embed, H * W)
        x = x.view((b, c, h * w))
        # (B, d_embed, H * W) -> (B, H * W, d_embed)
        x = x.transpose(-1, -2)
        # Self-attention WITHOUT mask
        # (B, H * W, d_embed) -> (B, H * W, d_embed)
        x = self.attention(x)
        # (B, H * W, d_embed) ->  (B, C, H * d_embed) 
        x = x.transpose(-1, -2)
        # (B, d_embed, H * W)  -> (B, d_embed, H , W) 
        x = x.view((b, c, h, w))
        #  (B, d_embed, H , W) -> (B, in_channels, H , W) 
        x = self.cnn2(x)
        # (B, in_channels, H , W) -> (B, in_channels, H , W)
        x = x + residual
        # (B, in_channels, H , W) -> # (B, H , W, in_channels)
        x = x.permute(0, 2, 3, 1)
        # (B, H , W, in_channels) -> (B, H , W, in_channels)
        x = self.norm(x) 
        # (B, H , W, in_channels) -> (B, in_channels, H , W)
        x = x.permute(0, 3, 1, 2)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=512,
                 height = 6,
                 width = 18,
                 out_channels=1024,
                 enc_unit=3,
                 n_heads=8):
        super().__init__()

        self.height = height
        self.width = width
        self.seq_len = height * width
        self.d_model = in_channels

        self.pos_encoder = PositionalEncoding(d_model=self.d_model,
                                              seq_len=self.seq_len)
        self.layers = nn.ModuleList([
            EncoderUnit(in_channels, out_channels, n_heads)
            for _ in range(enc_unit)
        ])

    def forward(self, x):
        # x: (B, in_channels, H, W)
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x = x.flatten(2).permute(0, 2, 1)
        # (B, H*W, C)
        x = self.pos_encoder(x)
        # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        for layer in self.layers:
            x = layer(x)
        return x

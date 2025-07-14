import torch
import torch.nn as nn
from feature_extractor import CNNBlock
from attention import SelfAttention
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        self.d_model = d_model
        self.height = height
        self.width = width

        pe = torch.zeros(d_model, height, width)

        d_model_half = d_model // 2  # metà per x, metà per y

        # div_term: (d_model_half // 2,)
        div_term = torch.exp(
            torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half)
        )  # (d_model_half//2,)

        # Posizioni
        pos_w = torch.arange(0, width).float()  # (W,)
        pos_h = torch.arange(0, height).float()  # (H,)

        # Positional encoding per asse x (larghezza)
        pe_x = torch.zeros(d_model_half, width)  # (C//2, W)
        pe_x[0::2, :] = torch.sin(div_term.unsqueeze(1) * pos_w.unsqueeze(0))  # (C//4, W)
        pe_x[1::2, :] = torch.cos(div_term.unsqueeze(1) * pos_w.unsqueeze(0))

        # Positional encoding per asse y (altezza)
        pe_y = torch.zeros(d_model_half, height)  # (C//2, H)
        pe_y[0::2, :] = torch.sin(div_term.unsqueeze(1) * pos_h.unsqueeze(0))  # (C//4, H)
        pe_y[1::2, :] = torch.cos(div_term.unsqueeze(1) * pos_h.unsqueeze(0))

        # Combina: pe_y (H) lungo W, pe_x (W) lungo H
        pe[:d_model_half, :, :] = pe_y.unsqueeze(2).expand(-1, height, width)
        pe[d_model_half:, :, :] = pe_x.unsqueeze(1).expand(-1, height, width)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class AddAndNorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels) # NOTE: e se facessimo Group Norm?

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # (B, in_channels, H , W) -> (B, in_channels, H , W)
        x = x + residual
        # (B, in_channels, H , W) -> # (B, H , W, in_channels)
        x = x.permute(0, 2, 3, 1)
        # (B, H , W, in_channels) -> (B, H , W, in_channels)
        x = self.norm(x) 
        # (B, H , W, in_channels) -> (B, in_channels, H , W)
        x = x.permute(0, 3, 1, 2)
        return x

class EncoderUnit(nn.Module):
    def __init__(self, in_channels: int, d_embed: int, n_heads: int):
        super().__init__()
        # NOTE: [!] se metto padding = 1 (come nel paper) le dimensioni di x aumentano di 2 sia di height che di width 
        self.cnn1 = CNNBlock(in_channels=in_channels, out_channels=d_embed, kernel_size=1, stride=1, padding=0)
        self.attention = SelfAttention(n_heads=n_heads, d_embed=d_embed)
        # NOTE: [!] se metto padding = 1 ((come nel paper)) le dimensioni di x aumentano di 2 sia di height che di width
        self.cnn2 = CNNBlock(in_channels=d_embed, out_channels=in_channels, kernel_size=1, stride=1, padding=0) 
        self.addnorm = AddAndNorm(in_channels)  

    def forward(self, x):
        # x: (B, in_channels, H, W)
        residual = x
        # (B, in_channels, H, W) -> (B, d_embed, H, W)
        x = self.cnn1(x)
        # Self-attention WITHOUT mask
        # (B, d_embed, H, W) -> (B, d_embed, H , W) 
        x = self.attention(x)
        #  (B, d_embed, H , W) -> (B, in_channels, H , W) 
        x = self.cnn2(x)
        #(B, in_channels, H , W) -> (B, in_channels, H , W)
        x = self.addnorm(x, residual)

        return x # (B, in_channels, H , W)


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
        self.d_model = in_channels

        self.pos_encoder = PositionalEncoding(d_model=self.d_model,
                                              height=self.height,
                                              width=self.width)
        
        self.layers = nn.ModuleList([
            EncoderUnit(in_channels, out_channels, n_heads)
            for _ in range(enc_unit)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        
        # (B, in_channels, H, W) -> (B, in_channels, H, W)
        x = self.pos_encoder(x)
        # (B, in_channels, H, W) -> (B, in_channels, H, W)
        for layer in self.layers:
            x = layer(x)
        return x # # (B, in_channels, H, W)


# if __name__ == "__main__":
#     batch_size = 1
#     input_channels = 512
#     input_height = 6
#     input_width = 18

#     dummy_input = torch.randn(batch_size, input_channels, input_height, input_width)
#     print(f"Dimensione dell'input dummy: {list(dummy_input.shape)}")

#     model = Encoder()
#     # print(f"Modello PDLPR creato:\n{model}")
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\nNumero totale di parametri addestrabili: {total_params}")
#     size_in_mb = total_params * 4 / 1024 / 1024  # 4 bytes per param (float32)
#     print(f"Model size: {size_in_mb:.2f} MB")
    

#     output_features = model(dummy_input)

#     expected_output_shape = (batch_size, 512, 6, 18)
#     print(f"\nDimensione reali dell'output: {output_features.shape}")
#     print(f"Dimensione attesa dell'output: {expected_output_shape}")

#     assert output_features.shape == expected_output_shape, \
#         f"Errore: la dimensione dell'output non corrisponde a quella attesa! " \
#         f"Ottenuto: {output_features.shape}, Atteso: {expected_output_shape}"

#     print("\nTest completato con successo: Le dimensioni dell'output corrispondono a quelle attese.")

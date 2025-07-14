import torch
import torch.nn as nn
from attention import SelfAttention

class FocusStructure(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input size: (B, 3, H, W)
        patch1 = x[:, :, 0::2, 0::2] # (B, C, H/2, W/2)
        patch2 = x[:, :, 0::2, 1::2] # (B, C, H/2, W/2)
        patch3 = x[:, :, 1::2, 0::2] # (B, C, H/2, W/2)
        patch4 = x[:, :, 1::2, 1::2] # (B, C, H/2, W/2)
        x = torch.cat([patch1, patch2, patch3, patch4], dim=1)  # (B, 12, H/2, W/2)
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        if out_channels < 32:
            num_groups = 4
        else:
            num_groups = 32 

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        # self.bn = nn.BatchNorm2d(out_channels)            # NOTE: prima era Batch norm  
        self.activation = nn.SiLU()                         # NOTE: LeakyReLU in the original paper 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Features, Height, width) 
        # NOTE:  order is inverted compared to the original paper
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class RESBLOCK(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.cnn1 = CNNBlock(channels, channels, kernel_size, stride, padding)
        self.cnn2 = CNNBlock(channels, channels, kernel_size, stride, padding)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.cnn1(x)
        out = self.cnn2(out)
        return out + residual

#Output_size = floor((Input_size + 2 * Padding - Kernel_size) / Stride) + 1
class ConvDownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1):
        super().__init__()
        self.conv = CNNBlock(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# class AttentionBlock(nn.Module):
#         def __init__(self, n_heads: int, d_embed: int):
#             super().__init__()
#             self.attn = SelfAttention(n_heads=n_heads, d_embed=d_embed)
#             self.bn = nn.BatchNorm2d(d_embed)  
#             self.activation = nn.SiLU()
        
#         def forward(self, x):
#             residual = x
#             # NOTE: devo aggiungere positional encoding?
#             x = self.attn(x)
#             x = self.bn(x)
#             x =  self.activation(x)
#             out = x + residual
#             return out

class IGFE(nn.Module):
    def __init__(self, in_channels: int=12, out_channels: int=512):
        super().__init__()
        mid_channels = out_channels // 2  # NOTE: Qui potremmo cambiare e mettere un valore più piccolo/grande
        self.focus = FocusStructure()
        self.res1 = RESBLOCK(channels=in_channels)
        # self.attn1 = AttentionBlock(n_heads=n_heads, d_embed=in_channels)
        self.res2 = RESBLOCK(channels=in_channels)
        self.ConvDown1 = ConvDownSampling(in_channels=in_channels, out_channels=mid_channels) 
        self.res3 = RESBLOCK(channels=mid_channels)
        # self.attn2 = AttentionBlock(n_heads=n_heads, d_embed=mid_channels)
        self.res4 = RESBLOCK(channels=mid_channels)
        self.ConvDown2 = ConvDownSampling(in_channels = mid_channels, out_channels=out_channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, 12, H/2, W/2) 
        x = self.focus(x)
        # (B, 12, H/2, W/2) -> (B, 12, H/2, W/2) 
        x = self.res1(x) 
        # (B, 12, H/2, W/2) -> (B, 12, H/2, W/2) 
        # (B, 12, H/2, W/2) -> (B, 12, H/2, W/2)
        x = self.res2(x)
        # (B, 12, H/2, W/2) -> (B, 256, H/4, W/4)
        x = self.ConvDown1(x) 
        # (B, 256, H/4, W/4) -> (B, 256, H/4, W/4)
        x = self.res3(x)
        # (B, 256, H/4, W/4) -> (B, 256, H/4, W/4)
        # (B, 256, H/4, W/4) -> (B, 256, H/4, W/4)
        x = self.res4(x)
        # (B, 256, H/4, W/4) -> (B, 512, H/8, W/8)
        x = self.ConvDown2(x) #
        
        return x # if input has shape (B, 3, 48, 144) -> output has shape (B, 512, 6, 18)


# if __name__ == "__main__":
#     input_height = 48
#     input_width = 144
#     input_channels = 3
#     batch_size = 1

#     dummy_input = torch.randn(batch_size, input_channels, input_height, input_width)
#     print(f"Dimensione dell'input dummy: {dummy_input.shape}")

#     igfe_model = IGFE()
#     print(f"Modello IGFE creato:\n{igfe_model}")
#     total_params = sum(p.numel() for p in igfe_model.parameters() if p.requires_grad)
#     print(f"\nNumero totale di parametri addestrabili: {total_params}")
#     size_in_mb = total_params * 4 / 1024 / 1024  # 4 bytes per param (float32)
#     print(f"Model size: {size_in_mb:.2f} MB")
    

#     output_features = igfe_model(dummy_input)

#     expected_output_shape = (batch_size, 512, int(input_height/8), int(input_width/8))
#     print(f"\nDimensione delle feature estratte dall'IGFE: {output_features.shape}")
#     print(f"Dimensione attesa dell'output: {expected_output_shape}")

#     assert output_features.shape == expected_output_shape, \
#         f"Errore: la dimensione dell'output non corrisponde a quella attesa! " \
#         f"Ottenuto: {output_features.shape}, Atteso: {expected_output_shape}"

#     print("\nTest completato con successo: Le dimensioni dell'output corrispondono a quelle attese.")

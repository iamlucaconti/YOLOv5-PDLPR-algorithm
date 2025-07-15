import torch
import torch.nn as nn
from feature_extractor import IGFE, CNNBlock
from encoder import Encoder
from decoder import Decoder

class PDLPR(nn.Module):
    def __init__(self, d_embed: int=512, d_cross: int=18, units: int=3,
                 n_heads: int=8, height: int = 6, width: int=18, num_classes: int=68):
        super().__init__()  

        self.igfe = IGFE(in_channels=12, out_channels=d_embed)

        self.encoder = Encoder(
            in_channels=d_embed,
            height = height,
            width = width,
            out_channels=d_embed*2,
            enc_unit=units,
            n_heads=n_heads)

        # self.cnn3 = CNNBlock(d_embed, d_embed, kernel_size=(2, 1), stride=(3, 1), padding=(1, 0))
        # self.cnn4 = CNNBlock(d_embed, d_embed, kernel_size=(1, 2), stride=(1, 3), padding=(0,0))
        self.cnn3 = CNNBlock(d_embed, d_embed, kernel_size=(3, 1), stride=(3, 1), padding=(1, 0))
        self.cnn4 = CNNBlock(d_embed, d_embed, kernel_size=(2, 1), stride=(1, 1), padding=(0,0))


        self.decoder = Decoder(
            height = height,
            width = width,
            d_embed=d_embed,
            d_cross=d_cross,
            dec_unit=units,
            n_heads=n_heads)
        
        # NOTE: ho aggiunto questo
        self.cnn5 = CNNBlock(d_embed, d_embed, kernel_size=(3, 1), stride=(3, 1), padding=(1, 0))
        self.cnn6 = CNNBlock(d_embed, d_embed, kernel_size=(2, 1), stride=(1, 1), padding=(0,0))

        self.classifier = nn.Linear(d_embed, num_classes)
    
    def forward(self, x):
        # (B, 3, 48, 144) -> (B, 512, 6, 18)
        x = self.igfe(x)
        # print("IGFE output:", x.shape)

        # (B, 512, 6, 18) -> (B, 512, 6, 18) 
        x = self.encoder(x)
        # print("Enc output:", x.shape)

        # (B, 512, 6, 18) -> (B, 512, 3, 6) 
        conv_out = self.cnn4(self.cnn3(x)) # (B, 512, 3, 6)

        # print("Output CNN3: ", self.cnn3(x).shape)
        # print("Output CNN4: ", self.cnn4(self.cnn3(x)).shape)
        # (B, 512, 6, 18) -> (B, 512, 6, 18)
        x = self.decoder(x, conv_out)
        # print("Dec output: ", x.shape)

        # (B, 512, 6, 18) -> (B, 512, 1, 18) 
        # print("Output CNN5: ", self.cnn5(x).shape)
        # print("Output CNN6: ", self.cnn6(self.cnn5(x)).shape)
        x = self.cnn6(self.cnn5(x))
        
        #  (B, 512, 1, 18) -> (B, 512, 18) -> (B, 18, 512) 
        x = x.squeeze(2).permute(0, 2, 1) 
        # (B, 18, 512) -> (B, 18, 68)
        logits = self.classifier(x)
        # print("Logits shape: ", logits.shape)
        return logits  # (B, 18, 68)




if __name__ == "__main__":
    batch_size = 1
    input_channels = 3
    input_height = 48
    input_width = 144

    dummy_input = torch.randn(batch_size, input_channels, input_height, input_width)
    print(f"Dimensione dell'input dummy: {list(dummy_input.shape)}")

    model = PDLPR()
    # print(f"Modello PDLPR creato:\n{model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNumero totale di parametri addestrabili: {total_params}")
    size_in_mb = total_params * 4 / 1024 / 1024  # 4 bytes per param (float32)
    print(f"Model size: {size_in_mb:.2f} MB")
    

    output_features = model(dummy_input)

    expected_output_shape = (batch_size, 18, 68)
    print(f"\nDimensione reali dell'output: {output_features.shape}")
    print(f"Dimensione attesa dell'output: {expected_output_shape}")

    assert output_features.shape == expected_output_shape, \
        f"Errore: la dimensione dell'output non corrisponde a quella attesa! " \
        f"Ottenuto: {output_features.shape}, Atteso: {expected_output_shape}"

    print("\nTest completato con successo: Le dimensioni dell'output corrispondono a quelle attese.")

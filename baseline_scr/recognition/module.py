import torch
import torch.nn as nn

class BaselineRecognizer(nn.Module):
    def __init__(self, num_chars=7, num_classes=68):
        super().__init__()
        self.num_chars = num_chars  

        self.cnn = nn.Sequential(
            # [B, 3, 48, 144] -> [B, 64, 48, 144]
            nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #[B, 64, 48, 144] -> [B, 64, 24, 72]
            nn.MaxPool2d(2, 2),  

            # [B, 64, 24, 72] -> [B, 128, 24, 72] 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [B, 128, 24, 72]  ->  [B, 128, 12, 36]
            nn.MaxPool2d(2, 2),

            # [B, 128, 12, 36] -> [B, 256, 12, 36]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # [B, 256, 12, 36] -> [B, 256, 12, 36]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # CNN output : [B, 256, 12, 36]
        self.rnn_input_size = 256 * 12  # Channels * Height
        self.sequence_length = 36       # Width

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=512,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )

        self.char_projection = nn.Linear(512 * 2, num_classes)

        # Fixed position in the sequence
        step = self.sequence_length // self.num_chars
        offset = step // 2
        self.register_buffer("output_positions", torch.tensor(
            [i * step + offset for i in range(num_chars)]
        ))

    def forward(self, x):
        x = self.cnn(x)                   # [B, 256, 12, 36]
        x = x.permute(0, 3, 1, 2)         # [B, 36, 256, 12]
        x = x.reshape(x.size(0), x.size(1), -1)  # [B, 36, 3072]

        rnn_out, _ = self.rnn(x)          # [B, 36, 1024]

        selected = rnn_out[:, self.output_positions, :]  # [B, num_chars, 1024]

        logits = self.char_projection(selected)  # [B, num_chars, num_classes]
        return logits

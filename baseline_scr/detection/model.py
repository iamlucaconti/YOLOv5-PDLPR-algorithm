import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from typing import Tuple


class LPDetectorFPN(nn.Module):
    """
    ResNet‑18 + FPN (P3 & P4) → GAP → MLP → (cx,cy,w,h)∈[0,1]
    """
    def __init__(self, backbone_name: str = "resnet18",
                 pretrained: bool = True, head_hidden: int = 256):
        super().__init__()

        backbone, _ = self._build_backbone(backbone_name, pretrained)

        # feature blocks
        self.stem   = nn.Sequential(*backbone[:6])
        self.layer3 = backbone[6]          # out 14×14×256
        self.layer4 = backbone[7]          # out  7×7×512

        # FPN (P3, P4)
        self.c3_lat = nn.Conv2d(256, 128, 1, bias=False)
        self.c4_lat = nn.Conv2d(512, 128, 1, bias=False)

        self.p3_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.p4_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )

        # head 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),                   # 256
            nn.Linear(256, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(head_hidden, 4),
            nn.Sigmoid()
        )

       
        for n, p in self.stem.named_parameters():
            if n.startswith("0"):           # conv1
                p.requires_grad = False

    @staticmethod
    def _build_backbone(name: str, pretrained: bool = True
                        ) -> Tuple[nn.Sequential, int]:
        if name == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1
                             if pretrained else None)
            layers = list(m.children())[:-2]   
            return nn.Sequential(*layers), 512
        elif name == "efficientnet_b0":
            m = tvm.efficientnet_b0(
                weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            return m.features, 1280
        else:
            raise ValueError(f"Backbone {name!r} non supportato")

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # B×3×224×224
        x  = self.stem(x)           # 28×28×128
        c3 = self.layer3(x)         # 14×14×256
        c4 = self.layer4(c3)        #  7×7×512

        p4 = self.c4_lat(c4)                        # 7×7×128
        p3 = self.c3_lat(c3) + F.interpolate(p4, size=c3.shape[-2:],
                                             mode="nearest")  # 14×14×128

        p3 = self.p3_conv(p3)       # 14×14×128
        p4 = self.p4_conv(p4)       #  7×7×128
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")

        feat = self.pool(torch.cat([p3, p4_up], dim=1))      # B×256×1×1
        return self.regressor(feat)                          # B×4 ∈ [0,1]

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Bloque básico: Conv → BN → ReLU → Conv → BN → ReLU
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Downsampling: MaxPool → DoubleConv
# ============================================================

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


# ============================================================
# Upsampling: ConvTranspose2d → concat → DoubleConv
# ============================================================

class Up(nn.Module):
    """
    Bloque de decodificador:
    ConvTranspose2d para duplicar tamaño → concat con encoder → DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # ConvTranspose2d: toma in_channels y produce out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # después de concatenar, los canales se duplican
        self.conv = DoubleConv(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x_decoder, x_encoder):
        x_decoder = self.up(x_decoder)   # reduce canales, duplica tamaño

        # padding si hay mismatch de 1 pixel
        diffY = x_encoder.size(2) - x_decoder.size(2)
        diffX = x_encoder.size(3) - x_decoder.size(3)

        if diffX != 0 or diffY != 0:
            x_decoder = F.pad(
                x_decoder,
                [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2]
            )

        # concatenación encoder ←→ decoder
        x = torch.cat([x_encoder, x_decoder], dim=1)

        return self.conv(x)



# ============================================================
# Capa final
# ============================================================

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


# ============================================================
# U-Net completa multiescala
# ============================================================

class UNetMultiScale(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()

        # Encoder
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder 
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)


        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)    # 64
        x2 = self.down1(x1) # 128
        x3 = self.down2(x2) # 256
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 1024

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return self.outc(x)

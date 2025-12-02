import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18_4ch(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        # Cargar pesos ImageNet
        self.base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Guardamos la capa original
        old_conv = self.base.conv1

        # Crear nueva capa con 4 canales de entrada
        self.base.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Inicializar los 3 primeros canales con pesos ImageNet
        with torch.no_grad():
            self.base.conv1.weight[:, :3, :, :] = old_conv.weight

            # Canal extra se inicializa copiando canal rojo (index 0)
            self.base.conv1.weight[:, 3:4, :, :] = old_conv.weight[:, 0:1, :, :]

        # Reemplazar clasificación final para 4 clases de daño
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

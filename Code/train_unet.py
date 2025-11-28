"""
Entrenamiento de la U-Net Multiescala.

Este script:
    - Carga los DataLoaders (imágenes multiescala + máscaras)
    - Define la función de pérdida BCE + Dice Loss
    - Entrena la U-Net multiescala
    - Evalúa en validación
    - Guarda el mejor modelo según Dice Score
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any

from datasetMultiscale import create_dataloaders
from model_unet import UNetMultiScale


# ============================================================
# Métrica DICE
# ============================================================

def dice_coef(pred, target, eps=1e-7):
    """ Calcula Dice Coefficient entre predicciones y máscaras GT. """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


# ============================================================
# Pérdida BCE + Dice Loss
# ============================================================

class BCEDiceLoss(nn.Module):
    """
    Loss combinada:
        α * BCE + β * DiceLoss
    """
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice = dice_coef(probs, targets)
        dice_loss = 1.0 - dice
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


# ============================================================
# Entrenamiento por época
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


# ============================================================
# Evaluación (sin gradientes)
# ============================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, masks)
            running_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(logits)
            dice = dice_coef(probs, masks)
            running_dice += dice.item() * imgs.size(0)

    return (
        running_loss / len(loader.dataset),
        running_dice / len(loader.dataset),
    )


# ============================================================
# Entrenamiento completo
# ============================================================

def train_unet_multiscale(
    num_epochs=20,
    batch_size=2,
    lr=1e-4,
    device_str="cuda",
    model_save_path: Path = Path("models/unet_multiscale.pth"),
) -> Dict[str, Any]:

    # Selección de GPU o CPU
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    # Modelo
    model = UNetMultiScale(in_channels=6, out_channels=1).to(device)

    # DataLoaders
    train_loader, val_loader = create_dataloaders(batch_size=batch_size)

    # Loss
    criterion = BCEDiceLoss()

    # Optimizador
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Crear carpeta para guardar modelo
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0

    # Loop de entrenamiento
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        # Guardar mejor modelo
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), model_save_path)
            print(f"  → Nuevo mejor modelo guardado (Dice = {best_dice:.4f})")

    return history


# ============================================================
# Main — Ejecutar entrenamiento desde consola
# ============================================================

if __name__ == "__main__":
    train_unet_multiscale(
        num_epochs=15,
        batch_size=1,
        lr=1e-4,
        device_str="cuda"
    )

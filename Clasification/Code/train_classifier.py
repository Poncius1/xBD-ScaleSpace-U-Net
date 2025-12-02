import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from dataset_clasification import ClassificationDataset
from resNet18_model import ResNet18_4ch


# ============================================================
# CONFIGURACIÓN
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "Clasification" / "dataset" / "classification_dataset.csv"

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 20
VAL_SPLIT = 0.2   # igual que segmentation (80/20)

MODELS_DIR = PROJECT_ROOT / "Clasification" / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================
# FUNCIÓN DE ENTRENAMIENTO
# ============================================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# FUNCIÓN DE VALIDACIÓN
# ============================================================

def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    samples = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            samples += labels.size(0)

    acc = correct / samples
    return total_loss / len(loader), acc


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():

    print("\n===========================================")
    print("  ENTRENANDO CLASIFICADOR RESNET18 (4ch)")
    print("===========================================\n")

    # 1. Dataset
    full_dataset = ClassificationDataset(CSV_PATH)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Total muestras: {len(full_dataset)}")
    print(f"Train: {len(train_set)}   |   Val: {len(val_set)}\n")

    # 2. Modelo
    model = ResNet18_4ch(num_classes=4).to(DEVICE)

    # 3. Optimización
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0
    best_model_path = MODELS_DIR / "best_resnet18_4ch.pth"

    # 4. Entrenamiento
    for epoch in range(1, EPOCHS + 1):

        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f" Train Loss: {train_loss:.4f}")
        print(f" Val Loss  : {val_loss:.4f}")
        print(f" Val Acc   : {val_acc*100:.2f}%")
        print("------------------------------------------------")

        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✔ Nuevo mejor modelo guardado → {best_model_path}\n")

    print("\nEntrenamiento finalizado.")
    print(f"Mejor accuracy de validación: {best_acc*100:.2f}%")
    print(f"Modelo final guardado en: {best_model_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    main()

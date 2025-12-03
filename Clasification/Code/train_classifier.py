import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, List
import pandas as pd

from dataset_clasification import ClassificationDataset
from resNet18_model import ResNet18_4ch


# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "Clasification" / "dataset" / "classification_dataset.csv"

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 30
VAL_SPLIT = 0.2
PATIENCE = 6
ETA_MIN = 1e-6

MODELS_DIR = PROJECT_ROOT / "Clasification" / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================
# TRAIN EPOCH
# ============================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer
) -> float:

    model.train()
    total_loss = 0.0

    first = True

    for imgs, labels in loader:

        if first:
            print(" ✔ Primer batch cargado correctamente.")
            first = False

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
# VALIDATION EPOCH
# ============================================================

def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float]:

    model.eval()
    total_loss = 0.0
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

    acc = correct / samples if samples > 0 else 0.0
    return total_loss / len(loader), acc


# ============================================================
# SPLIT MANUAL TRAIN/VAL
# ============================================================

def create_train_val_datasets(
    csv_path: Path,
    val_split: float
) -> Tuple[ClassificationDataset, ClassificationDataset, pd.DataFrame]:

    # Cargar dataframe completo una sola vez
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    num_samples = len(df)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size

    g = torch.Generator()
    g.manual_seed(42)
    perm: List[int] = torch.randperm(num_samples, generator=g).tolist()

    train_indices = perm[:train_size]
    val_indices = perm[train_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    train_dataset = ClassificationDataset(csv_path, augment=True)
    val_dataset = ClassificationDataset(csv_path, augment=False)

    train_dataset.df = train_df
    val_dataset.df = val_df

    return train_dataset, val_dataset, df


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    class_counts = df["label"].value_counts().sort_index()

    num_classes = 4
    counts = []
    for c in range(num_classes):
        counts.append(class_counts.get(c, 1))

    counts_tensor = torch.tensor(counts, dtype=torch.float32)
    inv_freq = 1.0 / counts_tensor
    weights = inv_freq / inv_freq.sum()

    return weights.to(DEVICE)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n===========================================")
    print("   ENTRENANDO CLASIFICADOR RESNET18 (4ch)")
    print("===========================================\n")

    # ---------- DATASETS ----------
    train_dataset, val_dataset, full_df = create_train_val_datasets(CSV_PATH, VAL_SPLIT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"Total muestras: {len(full_df)}")
    print(f"Train: {len(train_dataset)}   |   Val: {len(val_dataset)}\n")
    print(" → Cargando primer batch (puede tardar unos segundos)...")

    # ---------- MODELO ----------
    model = ResNet18_4ch(num_classes=4).to(DEVICE)

    # ---------- PÉRDIDA CON PESOS ----------
    class_weights = compute_class_weights(full_df)
    print(f"Pesos por clase (CE): {class_weights.detach().cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=ETA_MIN
    )

    best_acc = 0.0
    patience_counter = 0

    best_model_path = MODELS_DIR / "best_resnet18_4ch.pth"

    # ---------- LOOP ----------
    for epoch in range(1, EPOCHS + 1):

        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f" Train Loss: {train_loss:.4f}")
        print(f" Val Loss  : {val_loss:.4f}")
        print(f" Val Acc   : {val_acc*100:.2f}%")
        print(f" LR actual : {scheduler.get_last_lr()[0]:.6f}")
        print("------------------------------------------------")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✔ Nuevo mejor modelo guardado → {best_model_path}\n")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{PATIENCE}\n")

        if patience_counter >= PATIENCE:
            print("⛔ Early stopping activado. FIN DEL ENTRENAMIENTO.\n")
            break

    print("\n===========================================")
    print("   ENTRENAMIENTO FINALIZADO")
    print("===========================================")
    print(f"Mejor accuracy de validación: {best_acc*100:.2f}%")
    print(f"Modelo guardado en: {best_model_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    main()

import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from datasetMultiscale import MultiScaleDataset
from model_unet import UNetMultiScale


# ============================
# 1. Validar existencia de splits
# ============================

BASE = Path("DB_filtered")
SPLIT_TRAIN = BASE / "splits" / "train.csv"
SPLIT_VAL   = BASE / "splits" / "val.csv"

print("\n== Verificando splits ==")
print("train.csv existe :", SPLIT_TRAIN.exists())
print("val.csv existe   :", SPLIT_VAL.exists())

if not SPLIT_TRAIN.exists():
    raise FileNotFoundError("ERROR: No existe train.csv — ejecuta prepareDataset.py")


# ============================
# 2. Cargar DataFrames
# ============================

print("\n== Cargando DataFrame de entrenamiento ==")
df_train = pd.read_csv(SPLIT_TRAIN)
df_val   = pd.read_csv(SPLIT_VAL)

print("Columnas:", df_train.columns)
print("Total train:", len(df_train))
print("Total val:", len(df_val))


# ============================
# 3. Crear datasets
# ============================

train_dataset = MultiScaleDataset(df_train)
val_dataset   = MultiScaleDataset(df_val)

print("\nDataset OK — tamaños correctos.")
print("Train dataset length:", len(train_dataset))
print("Val dataset length  :", len(val_dataset))


# ============================
# 4. Probar lectura de una muestra
# ============================

print("\n== Probando lectura de una muestra ==")
img_tensor, mask_tensor = train_dataset[0]

print("img_tensor shape :", img_tensor.shape)   # (6, 1024, 1024)
print("mask_tensor shape:", mask_tensor.shape)  # (1, 1024, 1024)


# ============================
# 5. Probar DataLoader
# ============================

print("\n== Probando DataLoader ==")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

batch_img, batch_mask = next(iter(train_loader))

print("Batch img shape :", batch_img.shape)    # torch.Size([1, 6, 1024, 1024])
print("Batch mask shape:", batch_mask.shape)   # torch.Size([1, 1, 1024, 1024])


# ============================
# 6. Probar modelo U-Net
# ============================

print("\n== Probando modelo U-Net multiescala ==")

model = UNetMultiScale(in_channels=6, out_channels=1)

with torch.no_grad():
    out = model(batch_img)

print("Salida del modelo:", out.shape)         # torch.Size([1, 1, 1024, 1024])


print("\n==============================")
print("== TODO FUNCIONA CORRECTAMENTE ==")
print("==============================")

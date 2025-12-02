import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path

from config import DB_FILTERED_ROOT
from prepareDataset import load_multiscale_sample


# ============================================================
#   DATASET MULTIESCALA
# ============================================================

class MultiScaleDataset(Dataset):
    """
    Dataset PyTorch que usa un DataFrame con rutas a:
        - image (RGB)
        - mask
        - sigma1, sigma2, sigma4

    Cada muestra retorna:
        img_tensor : (6, H, W)
        mask_tensor: (1, H, W)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.repo_root = Path(__file__).resolve().parent.parent  # raíz del proyecto

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()

        # Convertir rutas relativas → absolutas
        for key in ["image", "mask", "sigma1", "sigma2", "sigma4"]:
            path = Path(row[key])
            if not path.is_absolute():
                row[key] = str((self.repo_root / path).resolve())

        return load_multiscale_sample(row)


# ============================================================
#   FUNCIONES AUXILIARES
# ============================================================

def load_splits():
    """
    Carga train.csv y val.csv desde DB_filtered/splits.
    """
    splits_dir = DB_FILTERED_ROOT / "splits"

    train_csv = splits_dir / "train.csv"
    val_csv   = splits_dir / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError("ERROR: train.csv no existe. Ejecuta prepareDataset.py")

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    return train_df, val_df



def create_dataloaders(batch_size=2, num_workers=0, pin_memory=True):
    """
    Crea DataLoaders de entrenamiento y validación.
    """

    train_df, val_df = load_splits()

    train_dataset = MultiScaleDataset(train_df)
    val_dataset   = MultiScaleDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader



# ============================================================
#   TEST BÁSICO
# ============================================================

if __name__ == "__main__":
    print("Probando MultiScaleDataset...")

    train_loader, val_loader = create_dataloaders(batch_size=1)

    batch_img, batch_mask = next(iter(train_loader))

    print("Batch img shape :", batch_img.shape)   # (1, 6, 1024, 1024)
    print("Batch mask shape:", batch_mask.shape)  # (1, 1, 1024, 1024)

    print("Dataset OK.")
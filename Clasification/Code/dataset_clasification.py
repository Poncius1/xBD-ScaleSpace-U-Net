import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image


class ClassificationDataset(Dataset):
    """
    Dataset para clasificación de daño xBD.

    Devuelve:
        img_4ch: tensor (4, 224, 224)
        label  : entero {0,1,2,3}

    - Usa PIL (rápido, estable)
    - Resize fijo a 224x224
    - Augment opcional (solo flip H)
    - Normalización Imagenet
    """

    def __init__(self, csv_path, augment=False):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        # Mezcla determinista
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.augment = augment
        self.size = (224, 224)

        # Normalización Imagenet
        self.mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        self.std  = np.array([0.229, 0.224, 0.225], dtype="float32")

    # --------------------------------------------------------
    # Loaders usando PIL (MUCHO más rápidos que cv2.imread)
    # --------------------------------------------------------
    def _load_rgb(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        img = img.resize(self.size, Image.Resampling.BILINEAR)
        return np.asarray(img, dtype="float32") / 255.0

    def _load_mask(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L")
        img = img.resize(self.size, Image.Resampling.NEAREST)
        return np.asarray(img, dtype="float32") / 255.0

    # --------------------------------------------------------
    def __len__(self):
        return len(self.df)

    # --------------------------------------------------------
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        rgb = self._load_rgb(row["rgb"])
        mask = self._load_mask(row["mask"])

        # ---------- Augment ligero ----------
        if self.augment:
            if np.random.rand() < 0.5:
                rgb = np.fliplr(rgb).copy()
                mask = np.fliplr(mask).copy()

        # ---------- Normalizar ----------
        rgb = (rgb - self.mean) / self.std

        # ---------- A tensores ----------
        rgb_t = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float32)
        mask_t = torch.tensor(mask[None, :, :], dtype=torch.float32)

        img_4ch = torch.cat([rgb_t, mask_t], dim=0)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return img_4ch, label

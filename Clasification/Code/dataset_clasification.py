import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd


class ClassificationDataset(Dataset):
    """
    Dataset para clasificación de daño a edificios usando:
        - Recorte RGB del edificio
        - Máscara de segmentación predicha (1 canal)
    
    Produce un tensor 4 canales: [R, G, B, Mask]
    """

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Mezclar ejemplos
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ---- Cargar imágenes ----
        rgb = cv2.imread(row["rgb"], cv2.IMREAD_COLOR)
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)

        if rgb is None or mask is None:
            raise FileNotFoundError(f"No se pudo cargar: {row['rgb']} o {row['mask']}")

        # ---- Convertir a RGB y normalizar ----
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype("float32") / 255.0
        mask = mask.astype("float32") / 255.0

        # ---- Convertir a tensores ----
        rgb_t = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float32)  # (3, H, W)
        mask_t = torch.tensor(mask[None, :, :], dtype=torch.float32)       # (1, H, W)

        # ---- Unir en un solo tensor 4 canales ----
        img_4ch = torch.cat([rgb_t, mask_t], dim=0)

        # ---- Etiqueta de daño ----
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return img_4ch, label

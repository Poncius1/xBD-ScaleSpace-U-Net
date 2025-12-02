import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

from config import (
    DB_FILTERED_ROOT,
    FILTERED_IMAGES_DIR,
    FILTERED_MASKS_DIR,
    FILTERED_SCALES_DIR,
)


SIGMA1_DIR = FILTERED_SCALES_DIR / "sigma1"
SIGMA2_DIR = FILTERED_SCALES_DIR / "sigma2"
SIGMA4_DIR = FILTERED_SCALES_DIR / "sigma4"


def build_dataset_entries():
    """
    Construye un DataFrame donde cada fila contiene:
        image, mask, sigma1, sigma2, sigma4

    Solo incluye entradas completas donde todos los archivos existen.
    """
    image_paths = sorted(FILTERED_IMAGES_DIR.glob("*.png"))
    entries = []
    missing = 0

    for img_path in image_paths:
        name = img_path.name

        mask_path = FILTERED_MASKS_DIR / name.replace(".png", "_mask.png")
        s1_path = SIGMA1_DIR / name
        s2_path = SIGMA2_DIR / name
        s4_path = SIGMA4_DIR / name

        if not (mask_path.exists() and s1_path.exists() and s2_path.exists() and s4_path.exists()):
            missing += 1
            continue

        entries.append({
            "image": str(img_path),
            "mask": str(mask_path),
            "sigma1": str(s1_path),
            "sigma2": str(s2_path),
            "sigma4": str(s4_path),
        })

    print("Total imágenes detectadas:", len(image_paths))
    print("Entradas válidas:", len(entries))
    print("Entradas descartadas:", missing)

    return pd.DataFrame(entries)


def split_and_save(df: pd.DataFrame, test_size=0.2, seed=42):
    """
    Divide el DataFrame en train/val y guarda los CSV en DB_filtered/splits.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    print("Train:", len(train_df), "| Validation:", len(val_df))

    out_dir = DB_FILTERED_ROOT / "splits"
    out_dir.mkdir(exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)

    print("Splits guardados en:", out_dir)
    return train_df, val_df



def load_multiscale_sample(row):
    """
    Carga una muestra multicanal lista para U-Net multiescala.

    Entradas (desde row, una fila del DataFrame):
        - row["image"]   : ruta a imagen RGB original
        - row["sigma1"]  : imagen suavizada con σ=1 (grayscale)
        - row["sigma2"]  : imagen suavizada con σ=2
        - row["sigma4"]  : imagen suavizada con σ=4
        - row["mask"]    : máscara GT binaria

    Retorna:
        img_tensor  : np.array de shape (6, H, W) con valores en [0,1]
        mask_tensor : np.array de shape (1, H, W) con valores en [0,1]
    """
    # RGB
    img_bgr = cv2.imread(row["image"], cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {row['image']}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0

    # Escalas gaussianas en gris
    s1 = cv2.imread(row["sigma1"], cv2.IMREAD_GRAYSCALE)
    if s1 is None:
        raise FileNotFoundError(f"Sigma1 not found: {row['sigma1']}")
    s2 = cv2.imread(row["sigma2"], cv2.IMREAD_GRAYSCALE)
    if s2 is None:
        raise FileNotFoundError(f"Sigma2 not found: {row['sigma2']}")
    s4 = cv2.imread(row["sigma4"], cv2.IMREAD_GRAYSCALE)
    if s4 is None:
        raise FileNotFoundError(f"Sigma4 not found: {row['sigma4']}")

    s1 = s1.astype("float32") / 255.0
    s2 = s2.astype("float32") / 255.0
    s4 = s4.astype("float32") / 255.0

    # Máscara
    mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {row['mask']}")
    mask = mask.astype("float32") / 255.0

    # Stack de canales: R, G, B, s1, s2, s4
    img_tensor = np.stack([
        img[:, :, 0],
        img[:, :, 1],
        img[:, :, 2],
        s1, s2, s4
    ], axis=0)

    mask_tensor = np.expand_dims(mask, axis=0)

    return img_tensor, mask_tensor


if __name__ == "__main__":
    df = build_dataset_entries()
    split_and_save(df)
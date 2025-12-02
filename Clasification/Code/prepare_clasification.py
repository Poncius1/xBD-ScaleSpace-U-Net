import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import shape, Polygon
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_IMAGES = PROJECT_ROOT / "DB" / "train_images_labels_targets" / "train" / "images"
TRAIN_TARGETS = PROJECT_ROOT / "DB" / "train_images_labels_targets" / "train" / "targets"

# --- IMPORTANTE ---
# Aquí deben estar las MÁSCARAS PURAS generadas por tu U-Net
PRED_MASKS_DIR = PROJECT_ROOT / "predictions" / "masks"   

OUT_DATASET_DIR = PROJECT_ROOT / "Clasificacion" / "dataset"
OUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DATASET_DIR / "classification_dataset.csv"


# ============================================================
# UTILIDAD: convertir polígono a bounding box
# ============================================================
def polygon_to_bbox(poly: Polygon):
    minx, miny, maxx, maxy = poly.bounds
    return int(minx), int(miny), int(maxx), int(maxy)


# ============================================================
# CREAR DATASET DE CLASIFICACIÓN
# ============================================================
def build_classification_dataset():

    entries = []

    target_files = sorted(TRAIN_TARGETS.glob("*.json"))

    print(f"\nProcesando {len(target_files)} archivos target...\n")

    for target_path in tqdm(target_files):

        image_id = target_path.stem.replace("_target", "")

        rgb_path = TRAIN_IMAGES / f"{image_id}.png"
        mask_path = PRED_MASKS_DIR / f"{image_id}.png"

        if not rgb_path.exists():
            print(f"[WARN] Falta RGB: {rgb_path}")
            continue

        if not mask_path.exists():
            print(f"[WARN] Falta máscara predicha (U-Net): {mask_path}")
            continue

        # cargar rgb + máscara predicha
        rgb = cv2.imread(str(rgb_path))
        pred_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if pred_mask is None:
            print(f"[WARN] máscara inválida: {mask_path}")
            continue

        H, W = pred_mask.shape

        # cargar anotaciones xBD
        data = json.loads(target_path.read_text())

        for feat in data["features"]["xy"]:

            damage = feat["properties"]["subtype"]
            if damage == "unclassified":
                continue

            damage_label = {
                "no-damage": 0,
                "minor-damage": 1,
                "major-damage": 2,
                "destroyed": 3
            }.get(damage, None)

            if damage_label is None:
                continue

            poly = shape(feat["geometry"])

            x1, y1, x2, y2 = polygon_to_bbox(poly)

            # Limitar box a la imagen
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            crop_rgb = rgb[y1:y2, x1:x2]
            crop_mask = pred_mask[y1:y2, x1:x2]

            if crop_rgb.size == 0:
                continue

            # rutas de salida
            out_rgb = OUT_DATASET_DIR / f"{image_id}_{x1}_{y1}_rgb.png"
            out_mask = OUT_DATASET_DIR / f"{image_id}_{x1}_{y1}_mask.png"

            cv2.imwrite(str(out_rgb), crop_rgb)
            cv2.imwrite(str(out_mask), crop_mask)

            entries.append({
                "rgb": str(out_rgb),
                "mask": str(out_mask),
                "label": damage_label
            })

    df = pd.DataFrame(entries)
    df.to_csv(CSV_PATH, index=False)

    print("\n=====================================================")
    print(" Dataset de CLASIFICACIÓN generado")
    print(" Total edificios:", len(df))
    print(" CSV:", CSV_PATH)
    print("=====================================================")


if __name__ == "__main__":
    build_classification_dataset()

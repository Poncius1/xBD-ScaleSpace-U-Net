import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =====================================================
# RUTAS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_FILTERED = PROJECT_ROOT / "DB_filtered"
IMAGES_DIR  = DB_FILTERED / "images"
LABELS_DIR  = DB_FILTERED / "labels"
TARGETS_DIR = DB_FILTERED / "targets"

PRED_MASKS_DIR = PROJECT_ROOT / "predictions" / "masks_filtered"

OUT_DATASET_DIR = PROJECT_ROOT / "Clasification" / "dataset"
OUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DATASET_DIR / "classification_dataset.csv"


# =====================================================
# VALIDACIONES SEGURAS
# =====================================================

def safe_crop(img, y1, y2, x1, x2):
    """Recorte seguro, siempre valida límites y contenido."""
    H, W = img.shape[:2]

    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None

    crop = img[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 8 or w < 8:  # tamaño mínimo para evitar PNG corruptos
        return None

    if np.isnan(crop).any():
        return None

    return crop.copy()  # evitamos stride negativo


def safe_write(path, img):
    """Guardado seguro de PNG (si falla, retornamos False)."""
    try:
        ok = cv2.imwrite(str(path), img)
        return bool(ok)
    except Exception:
        return False


# =====================================================
# NORMALIZACIÓN DE ETIQUETAS
# =====================================================

def normalize_damage(subtype_raw):
    subtype = subtype_raw.strip().lower().replace("_", "-")

    damage_map = {
        "no-damage": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3,
        "unclassified": None,
        "un-classified": None,
        "unknown": None,
        "none": None,
        "": None
    }

    return damage_map.get(subtype, None)


# =====================================================
# EXTRAER BBOX DESDE TARGET
# =====================================================

def get_bounding_boxes(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < 20:
            continue

        boxes.append((x, y, x + w, y + h))

    return boxes


# =====================================================
# MAIN
# =====================================================

def build_classification_dataset():

    entries = []
    json_files = sorted(LABELS_DIR.glob("*.json"))

    print(f"\nProcesando {len(json_files)} imágenes filtradas...\n")

    for json_path in tqdm(json_files):

        base = json_path.stem

        rgb_path   = IMAGES_DIR      / f"{base}.png"
        target_path = TARGETS_DIR    / f"{base}_target.png"
        pred_path  = PRED_MASKS_DIR  / f"{base}.png"

        if not rgb_path.exists() or not target_path.exists() or not pred_path.exists():
            continue

        rgb = cv2.imread(str(rgb_path))
        target_mask = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

        if rgb is None or target_mask is None or pred_mask is None:
            continue

        H, W = target_mask.shape

        boxes = get_bounding_boxes(target_mask)
        if len(boxes) == 0:
            continue

        data = json.loads(json_path.read_text())
        features = data.get("features", {}).get("lng_lat", [])

        # seleccionar etiqueta válida
        label_value = None
        for feat in features:
            raw = feat["properties"].get("subtype", "")
            label_value = normalize_damage(raw)
            if label_value is not None:
                break

        if label_value is None:
            continue

        # procesar cada edificio
        for (x1, y1, x2, y2) in boxes:

            crop_rgb  = safe_crop(rgb, y1, y2, x1, x2)
            crop_pred = safe_crop(pred_mask, y1, y2, x1, x2)

            if crop_rgb is None or crop_pred is None:
                continue

            out_rgb  = OUT_DATASET_DIR / f"{base}_{x1}_{y1}_rgb.png"
            out_mask = OUT_DATASET_DIR / f"{base}_{x1}_{y1}_mask.png"

            if not safe_write(out_rgb, crop_rgb):
                continue
            if not safe_write(out_mask, crop_pred):
                continue

            entries.append({
                "rgb": str(out_rgb),
                "mask": str(out_mask),
                "label": label_value
            })

    df = pd.DataFrame(entries)
    df.to_csv(CSV_PATH, index=False)

    print("\n✔ Dataset creado sin archivos corruptos.")
    print("✔ Total edificios recortados:", len(df))
    print("✔ CSV guardado en:", CSV_PATH)


if __name__ == "__main__":
    build_classification_dataset()

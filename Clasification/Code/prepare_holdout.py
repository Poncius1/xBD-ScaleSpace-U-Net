"""
prepare_holdout_classification.py
---------------------------------
Versión robusta estilo dataset_train pero adaptada al HOLDOUT.

Este script:

✔ Lee los JSON del holdout (polígonos + damage_level)
✔ Lee SOLO imágenes POST del holdout
✔ Lee las máscaras predichas por la U-Net (predictions/masks)
✔ Extrae bounding boxes desde los polígonos
✔ Recorta RGB + máscara_predicha para cada edificio
✔ Normaliza etiquetas a {0, 1, 2, 3}
✔ Maneja casos corruptos, vacíos y PNG inválidos
✔ Genera: Clasification/dataset/holdout/
✔ Produce un CSV final para validar el clasificador

CSV: rgb_path, mask_path, label
"""

import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ============================================================
# RUTAS CORRECTAS PARA TU REPO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Tus datos del HOLDOUT están aquí:
HOLD_DIR = PROJECT_ROOT / "DB" / "hold_images_labels_targets" / "hold"

HOLD_IMAGES_DIR  = HOLD_DIR / "images"
HOLD_TARGETS_DIR = HOLD_DIR / "labels"   

# máscaras predichas por  U-Net
PRED_MASKS_DIR   = PROJECT_ROOT / "predictions" / "masks"

# salida
OUT_DATASET_DIR  = PROJECT_ROOT / "Clasification" / "dataset" / "holdout"
OUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = PROJECT_ROOT / "Clasification" / "dataset" / "holdout_classification_dataset.csv"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def safe_crop(img, y1, y2, x1, x2):
    """Recorte seguro, evitando imágenes corruptas o vacías."""
    H, W = img.shape[:2]

    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None

    crop = img[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return None

    if np.isnan(crop).any():
        return None

    return crop.copy()


def safe_write(path, img):
    """Guardado robusto evitando corrupción PNG."""
    try:
        ok = cv2.imwrite(str(path), img)
        return bool(ok)
    except:
        return False


# ============================================================
# NORMALIZACIÓN DE DAÑO
# ============================================================

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


# ============================================================
# POLYGON → BBOX
# ============================================================

def parse_polygon_wkt(wkt):
    """Convierte un POLYGON WKT a matriz Nx2."""
    wkt = wkt.replace("POLYGON ((", "").replace("))", "")
    coords = []
    for pair in wkt.split(","):
        x, y = pair.strip().split(" ")
        coords.append([float(x), float(y)])
    return np.array(coords, dtype=np.float32)


def polygon_to_bbox(poly, img_w, img_h, margin=2):
    xs, ys = poly[:, 0], poly[:, 1]

    x1 = max(int(xs.min()) - margin, 0)
    x2 = min(int(xs.max()) + margin, img_w - 1)
    y1 = max(int(ys.min()) - margin, 0)
    y2 = min(int(ys.max()) + margin, img_h - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def build_holdout_classification_dataset():

    print("\n=== Construyendo dataset HOLDOUT ===")

    entries = []
    json_files = sorted(HOLD_TARGETS_DIR.glob("*.json"))

    print(f"JSON encontrados en labels/: {len(json_files)}\n")

    for json_path in tqdm(json_files):

        data = json.loads(json_path.read_text())

        img_name = data["metadata"]["img_name"]

        # Solo imágenes POST
        if "post" not in img_name.lower():
            continue

        base = Path(img_name).stem

        img_path  = HOLD_IMAGES_DIR / img_name
        mask_path = PRED_MASKS_DIR / f"{base}_mask.png"

        if not img_path.exists() or not mask_path.exists():
            print(f"[WARN] Falta archivo para {base}")
            continue

        rgb_loaded = cv2.imread(str(img_path))
        if rgb_loaded is None:
            print(f"[WARN] No se pudo cargar {img_path}")
            continue
        rgb_img = cv2.cvtColor(rgb_loaded, cv2.COLOR_BGR2RGB)

        pred_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            print(f"[WARN] No se pudo cargar máscara predicha {mask_path}")
            continue

        H, W = rgb_img.shape[:2]

        # Polígonos del JSON
        for feat in data["features"]["xy"]:

            raw_label = feat["properties"].get("subtype", "")
            label = normalize_damage(raw_label)

            if label is None:
                continue

            poly = parse_polygon_wkt(feat["wkt"])
            bbox = polygon_to_bbox(poly, W, H)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            crop_rgb  = safe_crop(rgb_img,  y1, y2, x1, x2)
            crop_pred = safe_crop(pred_mask, y1, y2, x1, x2)

            if crop_rgb is None or crop_pred is None:
                continue

            out_rgb  = OUT_DATASET_DIR / f"{base}_{x1}_{y1}_rgb.png"
            out_mask = OUT_DATASET_DIR / f"{base}_{x1}_{y1}_mask.png"

            if not safe_write(out_rgb, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)):
                continue
            if not safe_write(out_mask, crop_pred):
                continue

            entries.append({
                "rgb_path":  str(out_rgb),
                "mask_path": str(out_mask),
                "label":     label
            })

    df = pd.DataFrame(entries)
    df.to_csv(CSV_PATH, index=False)

    print("\n✔ Dataset HOLDOUT creado correctamente.")
    print("✔ Total edificios recortados:", len(df))
    print("✔ CSV guardado en:", CSV_PATH)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_holdout_classification_dataset()

"""
VISUALIZACIÓN FINAL (MEJORADA)
------------------------------

GT (izquierda) = polígonos reales desde JSON
PRED (derecha) = contornos segmentados por U-Net + clasificación de daño

✔ Mucho más realista
✔ Dibuja todos los edificios detectados por segmentación
✔ Usa los colores oficiales xBD
✔ No depende de polígonos perfectos
"""

import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# COLORES OFICIALES xBD (BGR)
# ============================================================
COLORS = {
    0: ( 82, 255,   0),   # No damage
    1: (  0, 255, 255),   # Minor
    2: (  0, 127, 255),   # Major
    3: (  0,   0, 255),   # Destroyed
}

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[2]

IMG_DIR   = ROOT / "DB" / "hold_images_labels_targets" / "hold" / "images"
JSON_DIR  = ROOT / "DB" / "hold_images_labels_targets" / "hold" / "labels"
MASK_DIR  = ROOT / "predictions" / "masks"

CSV_PRED  = ROOT / "Clasification" / "dataset" / "predictions_holdout" / "pred_results_holdout.csv"

OUT_DIR   = ROOT / "Clasification" / "dataset" / "predictions_holdout" / "full_images_comparison_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load prediction map (base_x1_y1 → pred_label)
# ============================================================
def load_prediction_map(csv_path):
    df = pd.read_csv(csv_path)
    pred_map = {}

    for _, row in df.iterrows():
        stem = Path(row["rgb_path"]).stem   # base_x1_y1_rgb
        key  = stem.replace("_rgb", "")     # base_x1_y1
        pred_map[key] = int(row["pred"])

    return pred_map


# ============================================================
# Read polygons from JSON (GT)
# ============================================================
def parse_polygon_wkt(wkt):
    wkt = wkt.replace("POLYGON ((", "").replace("))", "")
    pts = []
    for xy in wkt.split(","):
        x, y = xy.strip().split(" ")
        pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.int32)


# ============================================================
# Extract building contours from segmentation mask
# ============================================================
def get_segmented_buildings(mask):

    # Threshold
    th = (mask > 127).astype("uint8") * 255

    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = []
    for c in contours:
        if cv2.contourArea(c) < 40:
            continue

        x, y, w, h = cv2.boundingRect(c)
        results.append((x, y, x+w, y+h, c))

    return results


# ============================================================
# MAIN — GT vs Predicción usando segmentación U-Net
# ============================================================
def generate_comparison(new_images):

    pred_map = load_prediction_map(CSV_PRED)

    for base in new_images:

        img_path  = IMG_DIR / f"{base}.png"
        json_path = JSON_DIR / f"{base}.json"
        mask_path = MASK_DIR / f"{base}_mask.png"

        # Load RGB
        img_base = cv2.imread(str(img_path))
        if img_base is None:
            print(f"[WARN] No image: {img_path}")
            continue

        img_gt   = img_base.copy()
        img_pred = img_base.copy()

        # Load JSON (GT polygons)
        data = json.loads(json_path.read_text())

        # -------------------------------------------------------------
        # LEFT: GT Polygons
        # -------------------------------------------------------------
        for feat in data["features"]["xy"]:
            subtype = feat["properties"].get("subtype", "").lower()
            dmg_map = {
                "no-damage": 0,
                "minor-damage": 1,
                "major-damage": 2,
                "destroyed": 3,
            }
            if subtype not in dmg_map:
                continue

            label = dmg_map[subtype]
            poly = parse_polygon_wkt(feat["wkt"])

            cv2.polylines(
                img_gt,
                [poly.reshape(-1,1,2)],
                True,
                COLORS[label],
                3
            )

        # -------------------------------------------------------------
        # RIGHT: Segmentation + Classification
        # -------------------------------------------------------------
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        buildings = get_segmented_buildings(mask)

        for (x1, y1, x2, y2, contour) in buildings:

            key = f"{base}_{x1}_{y1}"

            if key not in pred_map:
                continue

            label = pred_map[key]

            cv2.polylines(
                img_pred,
                [contour],
                True,
                COLORS[label],
                3
            )

        # -------------------------------------------------------------
        # FINAL CANVAS (with titles)
        # -------------------------------------------------------------
        h, w, _ = img_gt.shape
        canvas = np.zeros((h + 80, w * 2, 3), dtype=np.uint8)

        cv2.putText(canvas, "Ground Truth",
                    (int(w*0.20), 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,255,255), 3)

        cv2.putText(canvas, "U-Net Segmentation + Classification",
                    (int(w*1.10), 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

        canvas[80:h+80, :w] = img_gt
        canvas[80:h+80, w:] = img_pred

        out_path = OUT_DIR / f"compare_{base}.png"
        cv2.imwrite(str(out_path), canvas)

        print(f"✔ Guardado: {out_path}")

    print("\n🎉 Comparaciones generadas con segmentación REAL.\n")

if __name__ == "__main__":
    generate_comparison([
        "hurricane-florence_00000214_post_disaster",
        "hurricane-michael_00000364_post_disaster",
        "santa-rosa-wildfire_00000108_post_disaster",
    ])

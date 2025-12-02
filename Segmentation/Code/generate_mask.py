
import json
import cv2
import numpy as np
from pathlib import Path
from config import FILTERED_LABELS_DIR, FILTERED_MASKS_DIR, FILTERED_IMAGES_DIR


def load_polygons_and_size(json_path: Path):
    """
    Lee los polígonos en coordenadas de imagen (features['xy']) y
    el tamaño (width, height) desde metadata del JSON.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    width = int(data["metadata"]["width"])
    height = int(data["metadata"]["height"])

    polygons = []
    for feat in data["features"]["xy"]:
        wkt = feat["wkt"]
        wkt = wkt.replace("POLYGON ((", "").replace("))", "")
        coords = []
        for pair in wkt.split(","):
            x, y = pair.strip().split(" ")
            coords.append([float(x), float(y)])
        polygons.append(np.array(coords, dtype=np.int32))

    return polygons, (width, height)


def create_mask(polygons, size):
    """
    Genera una máscara binaria a partir de una lista de polígonos (np.array Nx2)
    en un lienzo de tamaño (width, height).
    """
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygons) > 0:
        cv2.fillPoly(mask, polygons, 255)
    return mask


def generate_all_masks(
    labels_dir: Path = FILTERED_LABELS_DIR,
    masks_dir: Path = FILTERED_MASKS_DIR,
    images_dir: Path = FILTERED_IMAGES_DIR,
):
    """
    Recorre todos los JSON en labels_dir y genera una máscara
    en masks_dir para cada uno. El nombre de la máscara será
    <img_name>_mask.png basado en metadata["img_name"].
    """
    masks_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(labels_dir.glob("*.json"))
    print("Total JSON encontrados:", len(json_files))

    count = 0
    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)

        img_name = data["metadata"]["img_name"]
        img_path = images_dir / img_name

        if not img_path.exists():
            print("Imagen no encontrada para JSON:", json_path.name)
            continue

        polygons, (w, h) = load_polygons_and_size(json_path)
        mask = create_mask(polygons, (w, h))

        mask_name = img_name.replace(".png", "_mask.png")
        out_path = masks_dir / mask_name
        cv2.imwrite(str(out_path), mask)
        count += 1

    print("Máscaras generadas:", count)


if __name__ == "__main__":
    generate_all_masks()

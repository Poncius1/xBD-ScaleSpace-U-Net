from pathlib import Path

# Ruta base del proyecto (ajusta si es necesario)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


HOLD_IMAGES_DIR = PROJECT_ROOT / "DB" / "hold_images_labels_targets" / "hold" / "images"

# Carpeta donde están los datasets originales descomprimidos
DB_TRAIN_ROOT = PROJECT_ROOT / "DB" / "train_images_labels_targets" / "train"

# Carpeta donde guardas el subconjunto filtrado
DB_FILTERED_ROOT = PROJECT_ROOT / "DB_filtered"

# Subcarpetas dentro de DB_filtered
FILTERED_IMAGES_DIR = DB_FILTERED_ROOT / "images"
FILTERED_LABELS_DIR = DB_FILTERED_ROOT / "labels"
FILTERED_MASKS_DIR  = DB_FILTERED_ROOT / "masks"
FILTERED_SCALES_DIR = DB_FILTERED_ROOT / "scales"

# Eventos seleccionados (puedes modificarlos si quieres probar otros)
SELECTED_EVENTS = [
    "guatemala-volcano",
    "hurricane-michael",
]

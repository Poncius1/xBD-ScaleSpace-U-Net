from pathlib import Path

# =====================================================================
# Ruta base del repositorio (carpeta raíz que contiene DB/, DB_filtered/, Segmentation/, etc.)
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =====================================================================
# Directorios del dataset original xBD
# =====================================================================

DB_ROOT = PROJECT_ROOT / "DB"

# TRAIN (dataset original)
DB_TRAIN_ROOT = DB_ROOT / "train_images_labels_targets" / "train"

# HOLDOUT (dataset original)
HOLD_ROOT = DB_ROOT / "hold_images_labels_targets" / "hold"
HOLD_IMAGES_DIR = HOLD_ROOT / "images"
HOLD_LABELS_DIR = HOLD_ROOT / "labels"
HOLD_TARGETS_DIR = HOLD_ROOT / "targets"

# =====================================================================
# Directorios del dataset filtrado para segmentación
# =====================================================================

DB_FILTERED_ROOT = PROJECT_ROOT / "DB_filtered"

FILTERED_IMAGES_DIR = DB_FILTERED_ROOT / "images"
FILTERED_LABELS_DIR = DB_FILTERED_ROOT / "labels"
FILTERED_MASKS_DIR  = DB_FILTERED_ROOT / "masks"
FILTERED_SCALES_DIR = DB_FILTERED_ROOT / "scales"
FILTERED_SPLITS_DIR = DB_FILTERED_ROOT / "splits"
FILTERED_TARGETS_DIR = DB_FILTERED_ROOT / "targets"

# =====================================================================
# Selección de eventos para filtrar
# Modifícalo cuando quieras cambiar a huracanes, terremotos, incendios, etc.
# =====================================================================

SELECTED_EVENTS = [
    "hurricane-michael",
    "hurricane-florence",
]


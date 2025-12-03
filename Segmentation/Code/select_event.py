import shutil
from pathlib import Path
from config import (
    DB_TRAIN_ROOT,
    DB_FILTERED_ROOT,
    FILTERED_IMAGES_DIR,
    FILTERED_LABELS_DIR,
    FILTERED_TARGETS_DIR,
    SELECTED_EVENTS,
)


def filter_events(
    source: Path = DB_TRAIN_ROOT,
    dest_images: Path = FILTERED_IMAGES_DIR,
    dest_labels: Path = FILTERED_LABELS_DIR,
    dest_targets: Path = FILTERED_TARGETS_DIR,
    selected_events=None,
):
    """
    Filtra imágenes *_post_disaster.png y copia:

        - imagen post_disaster
        - su JSON asociado
        - su target correspondiente: *_post_disaster_target.png
    """

    if selected_events is None:
        selected_events = SELECTED_EVENTS

    source_images  = source / "images"
    source_labels  = source / "labels"
    source_targets = source / "targets"

    # Crear carpetas destino
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    dest_targets.mkdir(parents=True, exist_ok=True)

    # Limpiar destino
    for f in dest_images.glob("*"): f.unlink()
    for f in dest_labels.glob("*"): f.unlink()
    for f in dest_targets.glob("*"): f.unlink()

    total = 0
    missing_json = 0
    missing_target = 0

    print("\n=== Filtrando imágenes por evento ===\n")

    # Procesar solo imágenes post-disaster
    for img_path in source_images.glob("*_post_disaster.png"):

        # Filtrar por evento
        if not any(ev in img_path.name for ev in selected_events):
            continue

        total += 1

        base_name = img_path.stem   # ejemplo: hurricane-michael_00000539_post_disaster

        # ------------------------------------------------------------
        # 1) Copiar imagen
        # ------------------------------------------------------------
        shutil.copy(img_path, dest_images / img_path.name)

        # ------------------------------------------------------------
        # 2) Copiar JSON asociado
        # ------------------------------------------------------------
        json_name = base_name + ".json"
        json_path = source_labels / json_name

        if json_path.exists():
            shutil.copy(json_path, dest_labels / json_name)
        else:
            print(f"❗ JSON no encontrado: {json_name}")
            missing_json += 1

        # ------------------------------------------------------------
        # 3) Copiar TARGET correcto ( *_post_disaster_target.png )
        # ------------------------------------------------------------
        target_name = base_name + "_target.png"   # ← REGLA CORRECTA
        target_path = source_targets / target_name

        if target_path.exists():
            shutil.copy(target_path, dest_targets / target_name)
        else:
            print(f"❗ TARGET no encontrado: {target_name}")
            missing_target += 1

    print("\n--- Filtrado completado ---")
    print(f"Total imágenes copiadas: {total}")
    print(f"JSON faltantes: {missing_json}")
    print(f"Targets faltantes: {missing_target}")
    print(f"DB_filtered generado en: {DB_FILTERED_ROOT}\n")


if __name__ == "__main__":
    filter_events()

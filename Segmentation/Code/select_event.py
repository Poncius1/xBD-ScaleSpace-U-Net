import shutil
from pathlib import Path
from Segmentation.Code.config import DB_TRAIN_ROOT, DB_FILTERED_ROOT, FILTERED_IMAGES_DIR, FILTERED_LABELS_DIR, SELECTED_EVENTS


def filter_events(
    source: Path = DB_TRAIN_ROOT,
    dest_images: Path = FILTERED_IMAGES_DIR,
    dest_labels: Path = FILTERED_LABELS_DIR,
    selected_events=None,
):
    """
    Filtra imágenes post_desastre pertenecientes a ciertos eventos y
    copia tanto las imágenes como sus archivos JSON asociados a DB_filtered.

    Parámetros:
        source         : carpeta base del train (con subcarpetas images/ y labels/)
        dest_images    : carpeta destino para imágenes filtradas
        dest_labels    : carpeta destino para labels .json filtrados
        selected_events: lista de substrings que deben aparecer en el nombre del archivo
    """
    
    if selected_events is None:
        selected_events = SELECTED_EVENTS

    source_images = source / "images"
    source_labels = source / "labels"

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    # Limpiar destino (opcional)
    for f in dest_images.glob("*"):
        f.unlink()
    for f in dest_labels.glob("*"):
        f.unlink()

    total = 0
    missing_json = 0

    for img_path in source_images.glob("*_post_disaster.png"):
        if any(ev in img_path.name for ev in selected_events):
            total += 1

            # Copiar imagen
            shutil.copy(img_path, dest_images / img_path.name)

            # Copiar JSON correspondiente
            label_name = img_path.name.replace(".png", ".json")
            json_path = source_labels / label_name

            if json_path.exists():
                shutil.copy(json_path, dest_labels / label_name)
            else:
                print("NO EXISTE JSON PARA:", img_path.name)
                missing_json += 1

    print("Filtrado completado.")
    print("Total imágenes copiadas:", total)
    print("JSON faltantes:", missing_json)


if __name__ == "__main__":
    filter_events()

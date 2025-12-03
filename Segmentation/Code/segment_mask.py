import torch
import cv2
import numpy as np
from pathlib import Path
from model_unet import UNetMultiScale
from scale_space import generate_scale_space
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# USAR SOLO DB_FILTERED
# ============================================================

DB_FILTERED = PROJECT_ROOT / "DB_filtered"
FILTERED_IMAGES = DB_FILTERED / "images"   # *_post_disaster.png

# Salida: máscaras SOLO de estos eventos
OUT_MASKS = PROJECT_ROOT / "predictions" / "masks_filtered"
OUT_MASKS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "models" / "unet_multiscale_50.pth"


# ============================================================
# Cargar modelo
# ============================================================

def load_model(model_path: Path):
    model = UNetMultiScale(in_channels=6, out_channels=1)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# Preprocesamiento multiescala
# ============================================================

def prepare_multiscale(img_path):
    img_bgr = cv2.imread(str(img_path))

    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {img_path}")

    s1, s2, s4 = generate_scale_space(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0

    s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    s2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    s4 = cv2.cvtColor(s4, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0

    sample = np.stack([
        img_rgb[:, :, 0],
        img_rgb[:, :, 1],
        img_rgb[:, :, 2],
        s1, s2, s4
    ], axis=0)

    return sample


# ============================================================
# Generar máscaras SOLO para los eventos filtrados
# ============================================================

def generate_filtered_masks(model):

    images = sorted(FILTERED_IMAGES.glob("*.png"))
    print(f"\nGenerando máscaras para {len(images)} imágenes filtradas...\n")

    for img_path in tqdm(images):

        try:
            sample = prepare_multiscale(img_path)
        except Exception as e:
            print(f"⚠ Saltando {img_path.name}: {e}")
            continue

        tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        binary = (prob > 0.5).astype("uint8")

        out_path = OUT_MASKS / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), binary * 255)

    print("\n✔ COMPLETADO: masks_filtered generadas en /predictions/masks_filtered/\n")


# ============================================================

if __name__ == "__main__":
    print("Cargando modelo...")
    model = load_model(MODEL_PATH)
    generate_filtered_masks(model)

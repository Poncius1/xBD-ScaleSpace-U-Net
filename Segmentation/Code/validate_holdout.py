import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model_unet import UNetMultiScale
from scale_space import generate_scale_space  
from config import HOLD_IMAGES_DIR   

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Cargar modelo
# ============================================================

def load_model(model_path: Path):
    model = UNetMultiScale(in_channels=6, out_channels=1)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print(f"Modelo cargado: {model_path}")
    return model


# ============================================================
# 2. Preparar tensores multiescala SOLO PARA HOLDOUT
# ============================================================

def load_multiscale_holdout(img_path: Path):
    """Carga una imagen de holdout y genera σ1, σ2, σ4 en RAM."""
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer {img_path}")

    # Escalas
    s1, s2, s4 = generate_scale_space(img_bgr)

    # Normalizar
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    s2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    s4 = cv2.cvtColor(s4, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0

    # Ensamble multiescala
    sample = np.stack([
        img_rgb[:, :, 0],
        img_rgb[:, :, 1],
        img_rgb[:, :, 2],
        s1, s2, s4
    ], axis=0)

    return sample, img_rgb  # también regresamos RGB original para mostrar


# ============================================================
# 3. Extraer contornos
# ============================================================

def extract_contours(binary_mask):
    mask_uint8 = (binary_mask * 255).astype("uint8")
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# ============================================================
# 4. Guardar figura compuesta (probabilidad, máscara, contornos)
# ============================================================

def save_combined_figure(prob, binary, img_rgb, contours, out_path: Path, title: str):
    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=14)

    # Probabilidades
    plt.subplot(1, 3, 1)
    plt.title("Probabilidades")
    plt.imshow(prob, cmap="jet")
    plt.axis("off")

    # Máscara
    plt.subplot(1, 3, 2)
    plt.title("Máscara binaria")
    plt.imshow(binary, cmap="gray")
    plt.axis("off")

    # Contornos sobre imagen
    plt.subplot(1, 3, 3)
    plt.title("Contornos sobre imagen")
    plt.imshow(img_rgb)
    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        plt.plot(cnt[:, 0], cnt[:, 1], color="red", linewidth=1)
    plt.axis("off")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. Pipeline principal
# ============================================================

def run_inference_holdout(model, max_images=None):
    hold_dir = Path(HOLD_IMAGES_DIR)
    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)

    image_paths = sorted(list(hold_dir.glob("*.png")))
    if max_images:
        image_paths = image_paths[:max_images]

    print(f"Procesando {len(image_paths)} imágenes del holdout...")

    for img_path in image_paths:
        print(f"→ procesando {img_path.name}")

        # 1. Preparar tensores multiescala
        sample, img_rgb = load_multiscale_holdout(img_path)
        tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 2. Predicción
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # 3. Máscara binaria
        binary = (prob > 0.5).astype("uint8")

        # 4. Contornos
        contours = extract_contours(binary)

        # 5. Guardar figura combinada
        out_path = out_dir / f"{img_path.stem}.png"
        save_combined_figure(prob, binary, img_rgb, contours, out_path, img_path.name)

    print("\n✔ Completado: figuras guardadas en /predictions/")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    model = load_model(Path("models/unet_multiscale_30.pth"))
    run_inference_holdout(model, max_images=3)  # cambia a None para TODAS
 
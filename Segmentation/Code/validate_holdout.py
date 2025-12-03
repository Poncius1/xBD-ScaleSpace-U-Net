"""
validate_holdout.py
-------------------

Este script permite procesar SOLO las imágenes POST-disaster del holdout.

Funciones:
✔ Generar máscaras predichas por la U-Net  (para clasificación)
✔ Generar figuras 2×2 (imagen, prob, máscara, contornos)
✔ Elegir qué generar: solo máscaras, solo figuras o ambas

Configuración en la llamada:
    save_masks=True/False
    save_figures=True/False
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# IMPORTS 
from config import HOLD_IMAGES_DIR
from model_unet import UNetMultiScale
from scale_space import generate_scale_space

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
# 2. Cargar imagen + multiescala
# ============================================================

def load_multiscale_holdout(img_path: Path):
    """Carga imagen RGB y genera escalas gaussianas."""
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer {img_path}")

    # Escalas Gaussianas
    s1, s2, s4 = generate_scale_space(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    s2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    s4 = cv2.cvtColor(s4, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0

    # Tensor de 6 canales
    sample = np.stack([
        img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2],
        s1, s2, s4
    ], axis=0)

    return sample, img_rgb


# ============================================================
# 3. Contornos
# ============================================================

def extract_contours(binary_mask):
    """Extrae contornos exteriores de la máscara binaria."""
    mask_uint8 = (binary_mask * 255).astype("uint8")
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# ============================================================
# 4. Guardar figura 2×2
# ============================================================

def save_final_figure(prob, binary, img_rgb, contours, out_path: Path, title: str):
    """Genera figura profesional 2×2 (original, prob, máscara, contornos)."""

    plt.figure(figsize=(12, 10))
    plt.suptitle(title, fontsize=16)

    plt.subplot(2, 2, 1)
    plt.title("Imagen original POST")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Mapa de probabilidad")
    plt.imshow(prob, cmap="jet")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Máscara binaria predicha")
    plt.imshow(binary, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Contornos sobre imagen")
    plt.imshow(img_rgb)
    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        plt.plot(cnt[:, 0], cnt[:, 1], color="red", linewidth=1)
    plt.axis("off")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. Guardar máscara pura
# ============================================================

def save_mask(binary, out_path: Path):
    """Guarda máscara predicha."""
    cv2.imwrite(str(out_path), binary * 255)


# ============================================================
# 6. PIPELINE PRINCIPAL — SOLO POST
# ============================================================

def run_inference_holdout(
        model,
        max_images=None,
        selected_images=None,
        save_masks=True,
        save_figures=False):

    hold_dir = Path(HOLD_IMAGES_DIR)

    # Carpetas de salida
    out_masks = Path("predictions/masks")
    out_figs = Path("predictions/final_figures")

    if save_masks:
        out_masks.mkdir(parents=True, exist_ok=True)
    if save_figures:
        out_figs.mkdir(parents=True, exist_ok=True)

    # Obtener solo imágenes POST-del holdout
    all_paths = sorted([
        p for p in hold_dir.glob("*.png")
        if "post" in p.name.lower()
    ])

    print(f"Imágenes POST detectadas: {len(all_paths)}")

    # PRIORIDAD: selected_images > max_images > todas
    if selected_images:
        image_paths = [hold_dir / name for name in selected_images
                       if (hold_dir / name).exists() and "post" in name.lower()]
        print(f"Usando imágenes seleccionadas (POST): {len(image_paths)}")

    elif max_images:
        image_paths = all_paths[:max_images]
        print(f"Procesando {len(image_paths)} imágenes POST (max_images={max_images})")

    else:
        image_paths = all_paths
        print(f"Procesando TODAS las imágenes POST del holdout")

    # Procesamiento
    for img_path in image_paths:
        print(f"→ Procesando {img_path.name}")

        sample, img_rgb = load_multiscale_holdout(img_path)
        tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # inferencia U-Net
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        binary = (prob > 0.5).astype("uint8")
        contours = extract_contours(binary)

        # Guardar máscara
        if save_masks:
            mask_out = out_masks / f"{img_path.stem}_mask.png"
            save_mask(binary, mask_out)

        # Guardar figura final
        if save_figures:
            fig_out = out_figs / f"{img_path.stem}_final.png"
            save_final_figure(prob, binary, img_rgb, contours, fig_out, img_path.name)

    # Mensajes finales
    if save_masks:
        print("✔ Máscaras POST guardadas en → predictions/masks/")
    if save_figures:
        print("✔ Figuras POST guardadas en → predictions/final_figures/")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    model = load_model(Path("models/unet_multiscale_50.pth"))

    # OPCIONES:
    # ---------------------------------------
    # Solo máscaras (clasificación)
    # run_inference_holdout(model, save_masks=True, save_figures=False)

    # Solo figuras (reporte/presentación)
    #run_inference_holdout(model, max_images=5,save_masks=False, save_figures=True)

    # Ambas (pipeline completo)
    run_inference_holdout(
        model,
        selected_images=[
            "santa-rosa-wildfire_00000108_post_disaster.png",
        ],    
        save_masks=False,
        save_figures=True
    )

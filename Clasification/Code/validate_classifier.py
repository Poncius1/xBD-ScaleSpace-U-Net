"""
VALIDACIÓN HOLDOUT — VERSIÓN LIGERA Y ESTABLE
---------------------------------------------

Objetivo:
- Clasificación de los cuatro niveles de daño en HOLDOUT
- Calcular F1 macro, Cohen Kappa, MCC
- Guardar matriz de confusión
- Guardar SOLO N figuras de ejemplo (no de todos)

Diseño:
- Procesa 1 edificio a la vez (bajo consumo de RAM)
- No guarda todos los crops, solo mantiene una muestra aleatoria de num_examples
- Usa GPU si está disponible, si no, CPU
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import random
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, f1_score, cohen_kappa_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import sys

from resNet18_model import ResNet18_4ch


# ------------------------
# CONFIG GLOBAL
# ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
CLASSES = ["no-damage", "minor", "major", "destroyed"]

COLOR_GT = {0:(0,255,0),1:(255,215,0),2:(255,140,0),3:(255,0,0)}
COLOR_PRED = {0:(0,180,0),1:(200,170,0),2:(200,100,0),3:(180,0,0)}

IMAGENET_MEAN = np.array([0.485,0.456,0.406],dtype="float32")
IMAGENET_STD  = np.array([0.229,0.224,0.225],dtype="float32")


# ============================================================
# 1. CARGAR MODELO (CPU seguro, luego GPU si se puede)
# ============================================================
def load_model(model_path: Path) -> torch.nn.Module:
    if not model_path.exists():
        print(f"ERROR: no se encontró el modelo en: {model_path}")
        sys.exit(1)

    model = ResNet18_4ch(num_classes=4)

    # Cargamos SIEMPRE en CPU para evitar problemas raros de CUDA
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    global DEVICE
    try:
        model.to(DEVICE)
        print(f"✔ Modelo cargado en {DEVICE}")
    except RuntimeError:
        print("⚠ Error moviendo el modelo a GPU. Usando CPU.")
        DEVICE = torch.device("cpu")
        model.to(DEVICE)

    model.eval()
    return model


# ============================================================
# 2. CARGAR RGB + MASK (4 canales)
# ============================================================
def load_pair(rgb_path: str, mask_path: str):
    """
    Carga un crop RGB y su máscara, los reescala a 224x224 y devuelve:
    - img4: tensor numpy (4, H, W) normalizado
    - rgb_orig: imagen RGB [0,1] para visualización
    - mask: máscara [0,1] para visualización
    """
    # RGB
    rgb = Image.open(rgb_path).convert("RGB")
    rgb = rgb.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(rgb, dtype="float32") / 255.0

    # Normalizar como en entrenamiento (ImageNet)
    rgb_norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    rgb_norm = rgb_norm.transpose(2, 0, 1)  # (C,H,W)

    # Mask
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(IMG_SIZE, Image.Resampling.NEAREST)
    mask = np.asarray(mask, dtype="float32") / 255.0

    img4 = np.concatenate([rgb_norm, mask[None, :, :]], axis=0)
    return img4, rgb, mask


# ============================================================
# 3. FIGURA INDIVIDUAL PARA PRESENTACIÓN
# ============================================================
def save_visual(rgb, mask, gt, pred, probs, out_path: Path):
    """
    Crea una figura 2x2:
    - RGB con borde GT
    - Overlay máscara
    - RGB con borde Pred
    - Barras de probabilidades
    """
    plt.figure(figsize=(12,8))
    plt.suptitle(f"GT: {CLASSES[gt]} — Pred: {CLASSES[pred]}", fontsize=16)

    # RGB (0-1) → (0-255) uint8
    rgb_uint8 = (rgb * 255).astype("uint8")

    # (1) RGB con borde GT
    rgb_gt = rgb_uint8.copy()
    cv2.rectangle(rgb_gt, (0,0),(rgb_gt.shape[1]-1, rgb_gt.shape[0]-1), COLOR_GT[gt], 3)
    plt.subplot(2,2,1)
    plt.title("RGB (GT)")
    plt.imshow(rgb_gt)
    plt.axis("off")

    # (2) Overlay máscara
    mask_c = cv2.applyColorMap((mask*255).astype("uint8"), cv2.COLORMAP_JET)
    mask_c = cv2.cvtColor(mask_c, cv2.COLOR_BGR2RGB)
    overlay = 0.55*rgb + 0.45*(mask_c/255.0)
    plt.subplot(2,2,2)
    plt.title("Máscara Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    # (3) RGB con borde Pred
    rgb_pr = rgb_uint8.copy()
    cv2.rectangle(rgb_pr, (0,0),(rgb_pr.shape[1]-1, rgb_pr.shape[0]-1), COLOR_PRED[pred], 3)
    plt.subplot(2,2,3)
    plt.title("RGB (Pred)")
    plt.imshow(rgb_pr)
    plt.axis("off")

    # (4) Probabilidades
    plt.subplot(2,2,4)
    plt.title("Probabilidades por clase")
    plt.bar(CLASSES, probs, color=["green","gold","darkorange","red"])
    plt.ylim(0,1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# 4. MATRIZ DE CONFUSIÓN
# ============================================================
def save_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES
    )
    plt.title("Matriz de Confusión — Holdout")
    plt.xlabel("Predicción")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# 5. EVALUACIÓN COMPLETA (SIN REVENTAR RAM)
# ============================================================
def evaluate_classifier_holdout(model_path: Path, csv_path: Path, num_examples: int = 5):

    print("\n=== VALIDANDO CLASIFICADOR EN HOLDOUT (LIGERO) ===\n")

    model = load_model(model_path)

    if not csv_path.exists():
        print(f"ERROR: no existe el CSV del holdout: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Total de edificios en holdout: {total}\n")

    y_true, y_pred = [], []
    pred_rows = []

    # Para figuras: RESERVOIR SAMPLING (mantener solo num_examples en memoria)
    example_buffer = []  # lista de tuplas (rgb, mask, gt, pred, probs)
    seen = 0

    for idx, row in tqdm(df.iterrows(), total=total, desc="Validando"):

        rgb_path  = row["rgb_path"]
        mask_path = row["mask_path"]
        gt        = int(row["label"])

        img4, rgb, mask = load_pair(rgb_path, mask_path)

        x = torch.tensor(img4, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(np.argmax(probs))

        y_true.append(gt)
        y_pred.append(pred)

        pred_rows.append({
            "rgb_path": rgb_path,
            "mask_path": mask_path,
            "gt": gt,
            "pred": pred,
            "p0": probs[0], "p1": probs[1], "p2": probs[2], "p3": probs[3],
        })

        # -----------------------------
        # RESERVOIR SAMPLING PARA FIGURAS
        # -----------------------------
        seen += 1
        if len(example_buffer) < num_examples:
            # llenamos el buffer inicialmente
            example_buffer.append((rgb, mask, gt, pred, probs))
        else:
            # con prob (num_examples / seen) reemplazamos uno aleatorio
            j = random.randint(0, seen - 1)
            if j < num_examples:
                example_buffer[j] = (rgb, mask, gt, pred, probs)

        # Liberar tensores grandes explícitamente
        del x, logits
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ----- MÉTRICAS -----
    f1    = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)

    print("\n======= MÉTRICAS HOLDOUT =======")
    print(f"F1 macro : {f1:.4f}")
    print(f"Kappa    : {kappa:.4f}")
    print(f"MCC      : {mcc:.4f}")
    print("================================\n")

    # ----- SALIDA -----
    out_dir = csv_path.parent / "predictions_holdout"
    out_dir.mkdir(exist_ok=True)

    # CSV con todas las predicciones
    pred_csv_path = out_dir / "pred_results_holdout.csv"
    pd.DataFrame(pred_rows).to_csv(pred_csv_path, index=False)

    # Matriz de confusión
    cm_path = out_dir / "confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, cm_path)

    # Figuras para presentación
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    for i, (rgb, mask, gt, pred, probs) in enumerate(example_buffer):
        out_fig = fig_dir / f"example_{i}.png"
        save_visual(rgb, mask, gt, pred, probs, out_fig)

    print(f"✔ CSV de predicciones guardado en: {pred_csv_path}")
    print(f"✔ Matriz de confusión guardada en: {cm_path}")
    print(f"✔ Figuras de ejemplo guardadas en: {fig_dir}")

    return f1, kappa, mcc


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    model_path = ROOT / "models" / "best_resnet18_4ch.pth"
    csv_path   = ROOT / "Clasification" / "dataset" / "holdout_classification_dataset.csv"

    # Cambia num_examples si quieres 3, 5, 10 figuras para la presentación
    evaluate_classifier_holdout(model_path, csv_path, num_examples=5)

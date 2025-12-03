"""
VALIDACIÓN HOLDOUT — VERSIÓN FINAL ESTABLE
------------------------------------------

✔ Auto-GPU: usa GPU si hay VRAM; si no, fallback a CPU
✔ tqdm para progreso en consola
✔ Batches de 1 (seguro)
✔ Resize 224×224 (igual que en entrenamiento)
✔ Normalización ImageNet
✔ Guarda matriz de confusión
✔ Guarda figuras de ejemplo
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
# 1. CARGAR MODELO CON FALLBACK AUTOMÁTICO A CPU
# ============================================================
def load_model(model_path: Path):

    model = ResNet18_4ch(num_classes=4)

    # Siempre cargar pesos en CPU (seguro)
    state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state)

    global DEVICE
    try:
        model.to(DEVICE)
        print(f"✔ Modelo movido a {DEVICE}")
    except RuntimeError as e:
        print("⚠ No hay VRAM para cargar modelo en GPU.")
        print("→ Usando CPU automáticamente.")
        DEVICE = torch.device("cpu")
        model.to(DEVICE)

    model.eval()
    return model


# ============================================================
# 2. CARGAR RGB + MASK (4 canales)
# ============================================================
def load_pair(rgb_path, mask_path):

    # RGB
    rgb = Image.open(rgb_path).convert("RGB")
    rgb = rgb.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(rgb, dtype="float32") / 255.0

    rgb_norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    rgb_norm = rgb_norm.transpose(2, 0, 1)

    # Mask
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(IMG_SIZE, Image.Resampling.NEAREST)
    mask = np.asarray(mask, dtype="float32") / 255.0

    img4 = np.concatenate([rgb_norm, mask[None, :, :]], axis=0)
    return img4, rgb, mask


# ============================================================
# 3. FIGURA INDIVIDUAL
# ============================================================
def save_visual(rgb, mask, gt, pred, probs, out_path):

    plt.figure(figsize=(12,8))
    plt.suptitle(f"GT: {CLASSES[gt]} — Pred: {CLASSES[pred]}", fontsize=16)

    rgb_gt = (rgb*255).astype("uint8").copy()
    cv2.rectangle(rgb_gt, (0,0),(223,223), COLOR_GT[gt], 3)
    plt.subplot(2,2,1); plt.title("RGB (GT)"); plt.imshow(rgb_gt); plt.axis("off")

    mask_c = cv2.applyColorMap((mask*255).astype("uint8"), cv2.COLORMAP_JET)
    mask_c = cv2.cvtColor(mask_c, cv2.COLOR_BGR2RGB)
    overlay = 0.55*rgb + 0.45*(mask_c/255.0)
    plt.subplot(2,2,2); plt.title("Overlay"); plt.imshow(overlay); plt.axis("off")

    rgb_pr = (rgb*255).astype("uint8").copy()
    cv2.rectangle(rgb_pr, (0,0),(223,223), COLOR_PRED[pred], 3)
    plt.subplot(2,2,3); plt.title("RGB (Pred)"); plt.imshow(rgb_pr); plt.axis("off")

    plt.subplot(2,2,4)
    plt.title("Probabilidades")
    plt.bar(CLASSES, probs, color=["green","gold","darkorange","red"])
    plt.ylim(0,1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# 4. MATRIZ DE CONFUSIÓN
# ============================================================
def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Matriz de Confusión — Holdout")
    plt.xlabel("Predicción")
    plt.ylabel("GT")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# 5. EVALUACIÓN COMPLETA
# ============================================================
def evaluate_classifier_holdout(model_path, csv_path, num_examples=5):

    print("\n=== VALIDANDO CLASIFICADOR (MODO ESTABLE) ===\n")

    model = load_model(model_path)

    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Total de edificios: {total}\n")

    y_true, y_pred = [], []
    pred_rows = []
    example_pool = []

    # tqdm muestra barra de progreso
    for idx, row in tqdm(df.iterrows(), total=total, desc="Validando"):

        img4, rgb, mask = load_pair(row["rgb_path"], row["mask_path"])
        gt = int(row["label"])

        x = torch.tensor(img4, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(np.argmax(probs))

        y_true.append(gt)
        y_pred.append(pred)

        pred_rows.append({
            "rgb_path": row["rgb_path"],
            "mask_path": row["mask_path"],
            "gt": gt,
            "pred": pred,
            "p0": probs[0], "p1": probs[1], "p2": probs[2], "p3": probs[3],
        })

        example_pool.append((rgb, mask, gt, pred, probs))

    # ----- MÉTRICAS -----
    f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\n======= MÉTRICAS =======")
    print(f"F1 macro : {f1:.4f}")
    print(f"Kappa    : {kappa:.4f}")
    print(f"MCC      : {mcc:.4f}")
    print("=========================\n")

    # ----- SALIDA -----
    out_dir = csv_path.parent / "predictions_holdout"
    out_dir.mkdir(exist_ok=True)

    pd.DataFrame(pred_rows).to_csv(out_dir/"pred_results_holdout.csv", index=False)
    save_confusion_matrix(y_true, y_pred, out_dir/"confusion_matrix.png")

    # Figuras para presentación
    fig_dir = out_dir/"figures"
    fig_dir.mkdir(exist_ok=True)

    selected = random.sample(example_pool, min(num_examples, len(example_pool)))

    for i,(rgb,mask,gt,pred,p) in enumerate(selected):
        save_visual(rgb, mask, gt, pred, p, fig_dir/f"example_{i}.png")

    print(f"✔ Resultados guardados en: {out_dir}")

    return f1, kappa, mcc


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    model_path = ROOT / "models" / "best_resnet18_4ch.pth"
    csv_path   = ROOT / "Clasification" / "dataset" / "holdout_classification_dataset.csv"

    evaluate_classifier_holdout(model_path, csv_path, num_examples=5)

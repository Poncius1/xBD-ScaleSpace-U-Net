import random
import cv2

def apply_augmentation(img1, img2, img4, mask):
    """
    Recibe:
        img1, img2, img4 → 3 escalas BGR
        mask → máscara binaria 0/255

    Retorna:
        Las mismas imágenes transformadas.
    """

    # -----------------------------
    # 1. Flips horizontales/verticales
    # -----------------------------
    if random.random() < 0.5:
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)
        img4 = cv2.flip(img4, 1)
        mask = cv2.flip(mask, 1)

    if random.random() < 0.5:
        img1 = cv2.flip(img1, 0)
        img2 = cv2.flip(img2, 0)
        img4 = cv2.flip(img4, 0)
        mask = cv2.flip(mask, 0)

    # -----------------------------
    # 2. Rotación aleatoria
    # -----------------------------
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        def rot(x):
            return cv2.rotate(x, {90: cv2.ROTATE_90_CLOCKWISE,
                                  180: cv2.ROTATE_180,
                                  270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
        img1 = rot(img1)
        img2 = rot(img2)
        img4 = rot(img4)
        mask = rot(mask)

    # -----------------------------
    # 3. Brillo / Contraste (suave)
    # -----------------------------
    if random.random() < 0.3:
        alpha = 1 + (random.random() - 0.5) * 0.2   # contraste ±10%
        beta  = (random.random() - 0.5) * 25        # brillo ±25
        img1 = cv2.convertScaleAbs(img1, alpha=alpha, beta=beta)
        img2 = cv2.convertScaleAbs(img2, alpha=alpha, beta=beta)
        img4 = cv2.convertScaleAbs(img4, alpha=alpha, beta=beta)

    return img1, img2, img4, mask

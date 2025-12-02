import cv2
from pathlib import Path
from Segmentation.Code.config import FILTERED_IMAGES_DIR, FILTERED_SCALES_DIR


def generate_scale_space(img):
    """
    Aplica suavizado Gaussiano a una imagen BGR para generar
    tres versiones multi-escala.

    sigma = 1 → suavizado leve (detalles finos)
    sigma = 2 → suavizado medio (texturas generales)
    sigma = 4 → suavizado fuerte (formas globales)
    """
    img1 = cv2.GaussianBlur(img, (5, 5), sigmaX=1)
    img2 = cv2.GaussianBlur(img, (9, 9), sigmaX=2)
    img3 = cv2.GaussianBlur(img, (13, 13), sigmaX=4)
    return img1, img2, img3


def generate_scales_for_all(
    images_dir: Path = FILTERED_IMAGES_DIR,
    scales_root: Path = FILTERED_SCALES_DIR,
):
    """
    Genera las versiones sigma1, sigma2, sigma4 para cada imagen
    en images_dir y las guarda en:
        scales_root/sigma1
        scales_root/sigma2
        scales_root/sigma4
    """
    sigma1_dir = scales_root / "sigma1"
    sigma2_dir = scales_root / "sigma2"
    sigma4_dir = scales_root / "sigma4"

    for d in [scales_root, sigma1_dir, sigma2_dir, sigma4_dir]:
        d.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.png"))
    print("Total imágenes a procesar:", len(image_paths))

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print("No se pudo leer:", img_path)
            continue

        s1, s2, s4 = generate_scale_space(img)

        cv2.imwrite(str(sigma1_dir / img_path.name), s1)
        cv2.imwrite(str(sigma2_dir / img_path.name), s2)
        cv2.imwrite(str(sigma4_dir / img_path.name), s4)

    print("Generación de scale-space completada.")


if __name__ == "__main__":
    generate_scales_for_all()

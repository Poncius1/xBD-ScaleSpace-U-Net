# Proyecto de Segmentación con U-Net y Scale-Space (xBD)

## 🚀 Instalación rápida


Crear entorno virtual:

```bash
python -m venv venv
```

Activar entorno:

### Windows

```bash
venv\Scripts\activate
```


Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## 📂 Preparar datos

Crear estructura:

```bash
mkdir DB
mkdir DB_filtered
```

Colocar los datasets descargados en:

```
DB/
   train_images_labels_targets/
   test_images_labels_targets/
   hold_images_labels_targets/
```

---

## ▶️ Ejecutar notebooks

1. Filtrar eventos
   `Code/selectEvent.ipynb`

2. Generar máscaras y contornos
   `Code/generate_polygons.ipynb`

3. Generar escala-espacio
   `Code/scale_space.ipynb`

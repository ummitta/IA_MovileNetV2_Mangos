# Manifiesto compartido — MangoLeafBD v1 (1.000 imágenes)

Generado para la Issue #1: particiones únicas que consumirán las tres corridas
(MobileNetV2 base, MobileNetV2 ± Dropout y ResNet-18) sin recalcular nada.

## Datos de la generación

- **Dataset:** MangoLeafBD v1 — DOI [10.17632/hxsnvwty3r.1](https://doi.org/10.17632/hxsnvwty3r.1)
- **Imágenes fuente:** 4000 (500/clase)
- **Semilla:** `42`
- **Comando exacto:** `python scripts/build_manifest.py --dataset-dir data/raw/mangoleafbd --seed 42 --output data/manifests/mangoleaf_1000_manifest.csv`
- **SHA-256 del manifiesto:** `5d3901cca4bbc355e59714de8f021ed48e2fa258a5fd7024a22b13d602f8ce6e`
- **Filtro perceptual:** dHash 64 bits, 4 orientaciones, umbral 0
- **Versiones:** Python 3.13.5, Pillow 11.3.0

## Reparto por clase y partición

| class_idx | Clase | train | val | test | total |
| --- | --- | --- | --- | --- | --- |
| 0 | Anthracnose | 88 | 18 | 19 | 125 |
| 1 | Bacterial Canker | 88 | 18 | 19 | 125 |
| 2 | Cutting Weevil | 88 | 19 | 18 | 125 |
| 3 | Die Back | 88 | 19 | 18 | 125 |
| 4 | Gall Midge | 87 | 19 | 19 | 125 |
| 5 | Healthy | 87 | 19 | 19 | 125 |
| 6 | Powdery Mildew | 87 | 19 | 19 | 125 |
| 7 | Sooty Mould | 87 | 19 | 19 | 125 |
|  | **Total** | 700 | 150 | 150 | 1000 |

## Duplicados y solapamiento

- Copias exactas eliminadas antes del muestreo: **21**
  (21 hashes repetidos; 0 conflictos entre clases).
- Grupos near-duplicate: **3685**;
  con más de una imagen: **238**;
  tamaño máximo: **4**;
  grupos que cruzan clases: **0**.
- Señales que unieron grupos:

| Señal | Grupos |
| --- | --- |
| asset_stem | 226 |
| capture_timestamp | 233 |
| dhash | 232 |

## Identificadores de hoja (lo que se pudo comprobar)

- IDs de hoja publicados por los autores: **no disponibles**. Los nombres codifican el
  instante de captura (`YYYYMMDD_HHMMSS`) y el posprocesado (`(Custom)`, `- Copy`, `(n)`).
- Los autores declaran ~1.800 hojas distintas entre las 4.000 imágenes; el resto son
  variantes por zoom/rotación.
- Señales usadas: SHA-256 exacto, stem normalizado, timestamp de captura y dHash con
  rotaciones de 90°.
- **Alcance:** las copias exactas se eliminan por completo; las variantes re-guardadas,
  ampliadas y rotadas en múltiplos de 90° quedan agrupadas. Rotaciones arbitrarias o
  recortes agresivos **no son demostrables** con los metadatos disponibles y se declaran
  como riesgo residual.

## Integridad

- Imágenes válidas verificadas con Pillow: **4000**.
- Archivos inválidos: **0**.
- Filas del manifiesto: **1000** (125 por clase; 700/150/150).

## Riesgo residual

El dataset no publica un identificador de hoja. Se agruparon copias exactas (SHA-256), variantes re-guardadas con el mismo stem/timestamp y variantes rotadas en múltiplos de 90° o re-guardadas con dHash <= umbral. Rotaciones arbitrarias, recortes agresivos o cambios de escala que no conservan el nombre ni el dHash pueden escapar a la detección: el solapamiento entre particiones no puede demostrarse más allá de estas señales.

# IA_MovileNetV2_Mangos — clasificación de enfermedades en hojas de mango

Proyecto grupal de Inteligencia Artificial. El objetivo es comparar **MobileNetV2**
(con y sin regularización/Dropout) frente a **ResNet-18** usando **exactamente las
mismas particiones de datos** sobre el dataset [MangoLeafBD v1](https://doi.org/10.17632/hxsnvwty3r.1).

Este repositorio resuelve la **Issue #1**: adquisición reproducible del dataset y
generación de un manifiesto estratificado de 1.000 imágenes (125 por clase) con
particiones fijas `train=700`, `val=150`, `test=150`. Los pipelines de entrenamiento,
evaluación y benchmarking de inferencia (Issues #2–#6) consumirán el manifiesto sin
recalcular particiones.

## Dataset

- **MangoLeafBD v1** — Mendeley Data, DOI [10.17632/hxsnvwty3r.1](https://doi.org/10.17632/hxsnvwty3r.1)
  ([página del dataset](https://data.mendeley.com/datasets/hxsnvwty3r/1)).
- 4.000 imágenes JPG (500 por clase) de 240×320, 8 clases:

  | idx | clase | idx | clase |
  | --- | --- | --- | --- |
  | 0 | Anthracnose | 4 | Gall Midge |
  | 1 | Bacterial Canker | 5 | Healthy |
  | 2 | Cutting Weevil | 6 | Powdery Mildew |
  | 3 | Die Back | 7 | Sooty Mould |

- Licencia CC BY-NC 3.0. Los autores indican que ~1.800 imágenes corresponden a hojas
  distintas y el resto son variantes obtenidas ampliando o rotando la misma hoja.

El subconjunto de trabajo son **1.000 imágenes (125 por clase)**: un 25 % estratificado
del dataset original. Los archivos descargados viven en `data/raw/` y **nunca** se
versionan (ver `.gitignore`).

## Requisitos e instalación

Python ≥ 3.10 (probado con 3.13.5). Se recomienda un entorno virtual:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Dependencias de ejecución: `Pillow`, `requests`, `tqdm`. De desarrollo: `pytest`,
`ruff`, `black`.

## 1. Descarga del dataset (reproducible)

```bash
.venv/bin/python scripts/download_dataset.py
```

El script:

1. Descarga el ZIP oficial (`https://data.mendeley.com/public-api/zip/hxsnvwty3r/download/1`,
   ~108 MB) con reanudación por `Range` y comprobación de tamaño y CRC.
2. Extrae las 8 carpetas de clase en `data/raw/mangoleafbd/<Clase>/*.jpg` (sin el
   directorio raíz del ZIP).
3. Verifica el **SHA-256 de las 4.000 imágenes** contra los metadatos publicados por
   Mendeley (`content_details.sha256_hash`) y escribe `data/raw/mangoleafbd/.source.json`
   como marcador de procedencia.
4. Es idempotente: si el marcador y los conteos ya son correctos, no vuelve a descargar.

Opciones útiles: `--dest`, `--archive RUTA_DEL_ZIP` (si el servidor bloqueara la
descarga automática), `--metadata-json` (caché de metadatos) y `--skip-hash-check`.

## 2. Generación del manifiesto compartido

```bash
.venv/bin/python scripts/build_manifest.py \
  --dataset-dir data/raw/mangoleafbd \
  --seed 42 \
  --output data/manifests/mangoleaf_1000_manifest.csv
```

Salidas en `data/manifests/`:

- `mangoleaf_1000_manifest.csv` — una fila por imagen con
  `relative_path,class,class_idx,split,sha256,near_dup_group`.
- `mangoleaf_1000_manifest.meta.json` — semilla, comando exacto, versiones, fingerprint,
  tabla de conteos, estadísticas de duplicados y hallazgos sobre identificadores de hoja.
- `mangoleaf_1000_manifest.summary.md` — el mismo resumen en Markdown.
- `class_index.json` — mapeo fijo `class_idx` ↔ clase para todas las corridas.

Reparto determinista (semilla 42) — siempre el mismo, sin recálculo:

| idx | clase | train | val | test | total |
| --- | --- | --- | --- | --- | --- |
| 0 | Anthracnose | 88 | 18 | 19 | 125 |
| 1 | Bacterial Canker | 88 | 18 | 19 | 125 |
| 2 | Cutting Weevil | 88 | 19 | 18 | 125 |
| 3 | Die Back | 88 | 19 | 18 | 125 |
| 4 | Gall Midge | 87 | 19 | 19 | 125 |
| 5 | Healthy | 87 | 19 | 19 | 125 |
| 6 | Powdery Mildew | 87 | 19 | 19 | 125 |
| 7 | Sooty Mould | 87 | 19 | 19 | 125 |
| | **Total** | **700** | **150** | **150** | **1000** |

## 3. Verificación de integridad y reproducibilidad

```bash
# Verificación básica
.venv/bin/python scripts/verify_manifest.py \
  data/manifests/mangoleaf_1000_manifest.csv \
  --dataset-dir data/raw/mangoleafbd

# Verificación completa: rehash + decodificación Pillow + comparación con 2.ª corrida
.venv/bin/python scripts/verify_manifest.py \
  data/manifests/mangoleaf_1000_manifest.csv \
  --dataset-dir data/raw/mangoleafbd \
  --compare data/raw/repro/mangoleaf_1000_manifest.csv \
  --rehash --check-images
```

Las comprobaciones incluyen: esquema, 1.000 filas, 125 por clase, 700/150/150,
`class_idx` coherente, campos no vacíos, SHA-256 hexadecimal de 64 caracteres, rutas y
hashes únicos, intersecciones **vacías** de `relative_path`, `sha256` y `near_dup_group`
entre `train`/`val`/`test`, cada grupo near-duplicate en una sola partición, existencia y
contención de las rutas bajo `--dataset-dir`, rehash opcional, decodificación con Pillow,
coincidencia del fingerprint con los metadatos y comparación fila por fila con una
segunda corrida.

### Evidencia de reproducibilidad (ejecutada en este repositorio)

```console
$ .venv/bin/python scripts/build_manifest.py --dataset-dir data/raw/mangoleafbd \
    --seed 42 --output data/manifests/mangoleaf_1000_manifest.csv
...
CSV:      data/manifests/mangoleaf_1000_manifest.csv (1000 filas)
SHA-256 del manifiesto: 5d3901cca4bbc355e59714de8f021ed48e2fa258a5fd7024a22b13d602f8ce6e
Tiempo de generación: 10.5 s (seed=42)

$ .venv/bin/python scripts/build_manifest.py --dataset-dir data/raw/mangoleafbd \
    --seed 42 --output data/raw/repro/mangoleaf_1000_manifest.csv
SHA-256 del manifiesto: 5d3901cca4bbc355e59714de8f021ed48e2fa258a5fd7024a22b13d602f8ce6e

$ cmp data/manifests/mangoleaf_1000_manifest.csv data/raw/repro/mangoleaf_1000_manifest.csv
# (sin diferencias: idénticos byte a byte)

$ .venv/bin/python scripts/verify_manifest.py data/manifests/mangoleaf_1000_manifest.csv \
    --dataset-dir data/raw/mangoleafbd \
    --compare data/raw/repro/mangoleaf_1000_manifest.csv --rehash --check-images
RESULTADO GLOBAL: PASS (19 comprobaciones)
```

## 4. Cargadores compartidos para MobileNetV2 y ResNet-18

Con el dataset descargado y el entorno virtual activo, instala las dependencias de los
cargadores y recorre las tres particiones:

```bash
pip install -e ".[loaders]"
python scripts/inspect_loaders.py
python scripts/inspect_loaders.py --full
```

En Linux con CPU puedes instalar primero las ruedas CPU con
`pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu`.
El comando de inspección acepta `--dataset-dir`, `--manifest`, `--batch-size` y `--seed`.
Con `--full` comprueba los conteos 700/150/150, las ocho etiquetas por partición, las
dimensiones, los valores finitos y la repetibilidad de un ejemplo de validación.

```python
from mangoleaf.config import DEFAULT_MANIFEST_PATH, DEFAULT_RAW_DIR, IDX_TO_CLASS
from mangoleaf.loaders import build_loaders

loaders = build_loaders(DEFAULT_RAW_DIR, DEFAULT_MANIFEST_PATH, batch_size=32, seed=42)
images, labels, paths = next(iter(loaders["val"]))
print(images.shape, paths[0], IDX_TO_CLASS[labels[0].item()])
```

Cada lote entrega imágenes RGB `float32` en orden `(N, C, H, W)`, índices de clase
`int64` y rutas relativas en el mismo orden. Se conserva el último lote. Los tres
cargadores usan el índice fijo y la columna `split` del manifiesto. Validación y prueba
siguen el orden del CSV.

Se eligieron los pesos `MobileNet_V2_Weights.IMAGENET1K_V1` y
`ResNet18_Weights.IMAGENET1K_V1`. Ambos usan el mismo preprocesamiento de TorchVision:
redimensionar el lado corto a 256 (bilineal), recortar el centro a 224 × 224, convertir
los píxeles a [0, 1] y normalizar cada canal RGB mediante `(x - media) / desviación`.
Las medias son (0.485, 0.456, 0.406) y las desviaciones (0.229, 0.224, 0.225).
El cargador aplica el escalado del artículo una sola vez y entrega el tensor ya
normalizado al modelo.

Solo en entrenamiento se mezcla el orden y se aplican tres decisiones independientes
antes del preprocesamiento común: volteo horizontal (50 %), rotación y zoom conjuntos
(50 %, ángulo uniforme entre −10° y +10° y escala entre 1.0 y 1.1), y cambio de brillo
(50 %, factor uniforme entre 0.85 y 1.15). Para comparar arquitecturas, crea los
cargadores al inicio de cada corrida con la misma semilla (42 por defecto) y recorre
los lotes en la misma secuencia. El muestreo y los aumentos tienen generadores propios.

## Estructura del repositorio

```text
.
├── data/
│   ├── manifests/                 # artefactos versionados (CSV, meta, resumen, índice)
│   └── raw/                       # dataset descargado (IGNORADO por git)
├── scripts/
│   ├── _bootstrap.py              # añade src/ al sys.path
│   ├── download_dataset.py        # descarga + verificación SHA-256
│   ├── build_manifest.py          # muestreo estratificado determinista
│   └── verify_manifest.py         # comprobaciones de integridad/fuga/reproducibilidad
├── src/mangoleaf/
│   ├── config.py                  # clases, reparto exacto, rutas y semilla
│   ├── dataset.py                 # escaneo y validación con Pillow
│   ├── dedup.py                   # SHA-256, stems/timestamps y dHash rotacional
│   ├── build.py                   # selección por clase y asignación de particiones
│   ├── manifest.py                # lectura/escritura y resumen del manifiesto
│   └── verify.py                  # motor de verificación reutilizable
└── tests/                         # suite pytest (config, dataset, dedup, build, manifest, verify)
```

## Duplicados, fuga de datos y límites

Todo se ejecuta **antes** del muestreo:

1. **Copias exactas**: se agrupan por SHA-256 y se conserva un único representante
   (prefiriendo el nombre de archivo más limpio). En MangoLeafBD v1 se eliminan 21
   copias exactas, todas dentro de su misma clase (14 en Anthracnose, 7 en Die Back).
2. **Variantes re-guardadas o renombradas**: se agrupan por *stem* normalizado
   (se quitan `- Copy`, `(Custom)`, `(n)` y el prefijo `IMG_`) y por *timestamp* de
   captura `YYYYMMDD_HHMMSS` dentro de la misma clase.
3. **Variantes rotadas/re-codificadas**: se agrupan por dHash de 64 bits comparado en
   las 4 orientaciones de 90°. El umbral por defecto es **0** (coincidencia exacta bajo
   alguna rotación).
4. **Un representante por grupo** entra en el muestreo y los miembros de un grupo
   *nunca* se reparten entre particiones: si un grupo se elige, sus variantes quedan
   juntas en el mismo split (o no se eligen).

Decisiones y límites documentados en `mangoleaf_1000_manifest.meta.json`:

- Los autores **no publican un identificador de hoja**. Los nombres solo codifican el
  instante de captura y el posprocesado.
- Los grupos near-duplicate **no cruzan clases**: las particiones se estratifican por
  clase, así que un grupo mixto no podría asignarse a un único split. Las copias
  exactas con etiquetas distintas se reportan como conflicto (`cross_class_conflicts`).
- El umbral dHash se fijó en 0 porque valores mayores producen cadenas transitivas
  (single-linkage) entre hojas visualmente parecidas: con umbral 6, Die Back quedaba
  con solo 107 grupos y era imposible alcanzar 125 imágenes por clase. El umbral es
  configurable con `--dhash-threshold`.
- Rotaciones arbitrarias, recortes agresivos o cambios de escala que no conserven el
  nombre ni el dHash pueden escapar a la detección. El solapamiento residual entre
  particiones no es demostrable con los metadatos disponibles; queda declarado como
  riesgo residual en el resumen generado.

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .               # estilo/imports
.venv/bin/black --check .            # formato
```

## Flujo de trabajo

- La rama de trabajo de la Issue #1 es `feat/issue-1-dataset-manifest`.
- Nunca se hace push directo a `main`.
- Los artefactos de `data/manifests/` se versionan; `data/raw/`, imágenes y ZIPs no.

## Issue #3 · MobileNetV2 con y sin Dropout

Desde la raíz del repositorio y con el entorno Python activo:

```bash
python -m pip install -e '.[loaders,dev]'
python scripts/download_dataset.py
python -m pytest
python scripts/train_mobilenet.py --smoke --output runs/mobilenet-smoke
python scripts/train_mobilenet.py --output runs/mobilenet-completo
```

Cada comando de entrenamiento ejecuta **ambas variantes**. Usa una carpeta de salida nueva
para cada repetición. Se selecciona CUDA cuando está disponible y CPU en los demás casos.
Se puede indicar el dispositivo con `--device` y ajustar los hilos de CPU con `--threads`.
La prueba rápida usa un lote por fase en entrenamiento y validación.

Las dos variantes copian los mismos pesos ImageNet V1 y la misma cabeza inicial de ocho
clases. Solo cambia la probabilidad del Dropout existente en torchvision (0,5 o 0).
Cada corrida reinicia la semilla 42 y los cargadores de la issue #2. La selección usa
las 150 imágenes de validación y reserva las 150 de test para la evaluación posterior.

| Fase | Épocas | LR inicial | Parámetros actualizados |
| --- | ---: | ---: | --- |
| Base congelada | 10 | 0,0001 | Cabeza; BatchNorm del extractor en evaluación |
| Ajuste fino | Hasta 20 | 0,00001 | Red completa |

Se usa entropía cruzada, lote 32 y un Adam nuevo al comienzo de cada fase.
Los contadores se reinician al cambiar de fase. Tras cinco épocas seguidas sin un nuevo
mínimo de pérdida de validación, se multiplica la tasa por 0,5. La parada temprana
ocurre tras diez épocas seguidas sin mejora. El ajuste fino continúa desde los pesos al
final de la primera fase. El mejor checkpoint se selecciona entre **todas** las épocas
de ambas fases y, en caso de empate, se conserva el primero.

Cada variante escribe `config.json`, `history.csv` y `best.pt`. El directorio común
contiene `class_index.json` y `comparison.json`, que registra el mínimo de validación
y su época para cada variante. El script comprueba la salida de ocho logits, verifica
el peso de la primera convolución al entrenar en cada fase y recarga ambos checkpoints.
El [notebook de arquitectura](notebooks/mobilenetv2.ipynb) muestra las dimensiones
medidas y verifica la suma residual del bloque invertido.

## Issue #4 · ResNet-18 como referencia

El mismo script admite `--model resnet18`. Usa los pesos
`ResNet18_Weights.IMAGENET1K_V1` de torchvision y reemplaza `fc` por una capa
lineal de 512 entradas y ocho logits con sesgo. Esta cabeza carece de Dropout.
Con la base congelada, Adam ajusta los 4.104 parámetros de la cabeza. En el
ajuste fino quedan habilitados los 11.180.616 parámetros de la red adaptada.

Desde la raíz del repositorio, con el entorno virtual activo:

```bash
python -m pip install -e '.[loaders,dev]'
python scripts/download_dataset.py
python scripts/train_mobilenet.py --model resnet18 --smoke --output runs/resnet18-smoke
python scripts/train_mobilenet.py --model resnet18 --output runs/resnet18-completa
```

Usar un directorio de salida nuevo por ejecución. Para reproducir el entorno de
entrenamiento en CPU con cuatro hilos, añadir `--device cpu --threads 4`.
Se usan el manifiesto compartido, los aumentos y las imágenes RGB de 224 × 224
de la issue #2. Las dos fases reutilizan el protocolo de la issue #3, con
lote 32, semilla 42, Adam y entropía cruzada. La selección compara la pérdida
media de las 150 imágenes de validación en todas las épocas. Las 150 imágenes
de prueba quedan reservadas para la evaluación final.

El resultado completo queda en `runs/resnet18-completa/`. La raíz contiene
el índice de clases y `comparison.json` con la fase y época seleccionadas.
El subdirectorio `resnet18/` contiene `best.pt`, `history.csv` y `config.json`,
que incluye los parámetros habilitados en cada fase. La rutina comprueba la
inclusión de la cabeza en Adam y reproduce los logits de un ejemplo al
recargar el mejor checkpoint.

Para reconstruir la arquitectura, cargar los pesos y predecir la clase de la
primera imagen de validación, ejecutar desde el mismo entorno:

```bash
python - <<'PY'
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from mangoleaf.config import DEFAULT_MANIFEST_PATH, DEFAULT_RAW_DIR
from mangoleaf.loaders import build_loaders

run = Path("runs/resnet18-completa")
classes = json.loads((run / "class_index.json").read_text())["classes"]
names = {item["class_idx"]: item["class"] for item in classes}
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(names))
model.load_state_dict(torch.load(run / "resnet18/best.pt", map_location="cpu", weights_only=True))
model.eval()
images, labels, paths = next(iter(build_loaders(DEFAULT_RAW_DIR, DEFAULT_MANIFEST_PATH)["val"]))
with torch.inference_mode():
    predicted = model(images[:1]).argmax(dim=1).item()
print(paths[0], "predicción:", names[predicted], "etiqueta:", names[labels[0].item()])
PY
```

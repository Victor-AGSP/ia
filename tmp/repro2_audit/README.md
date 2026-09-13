# Proyecto 1 — Segmentación y clasificación de madurez de tomates (INFO1185)

Segmentación del fruto con K-Means (7 combinaciones de canales RGB) y clasificación
maduro/inmaduro con clasificadores bayesianos gaussianos (selección por análisis, SFS y PCA).

## Estructura

```
tomato_project.py        orquestador: ejecuta todo el experimento y genera results/
tomato/config.py         TODOS los parámetros y semillas
tomato/data.py           carga, control de duplicados, partición estratificada por imagen
tomato/segmentation.py   K-Means, elección del cluster fruto, post-procesado, Jaccard
tomato/features.py       descriptores de color (RGB, H circular, S, V, a*, b*)
tomato/classification.py razón de verosimilitudes, Youden, SFS, PCA, bootstrap
tomato/plots.py          figuras
tomato/reporting.py      tablas LaTeX y macros generadas desde los resultados
report.tex               informe (usa results/tex/*.tex; no contiene números escritos a mano)
presentation/            generador de la presentación PowerPoint
data/                    imágenes y máscaras (ripe/, unripe/ con images/ y masks/)
```

## Reproducir

```bash
pip install -r requirements.txt
python -X utf8 tomato_project.py            # ~3-5 min; --quick para prueba rápida
pdflatex report.tex && pdflatex report.tex  # informe PDF
cd presentation && npm install && node build_deck.js   # presentación .pptx
```

`--data` acepta la carpeta del dataset (por defecto busca `data/` y luego `dataset/`).

## Protocolo (resumen)

* Semilla global `20260913`. Antes de particionar se detectan (MD5) y eliminan 2 pares de
  imágenes idénticas, para que una copia no pueda quedar en entrenamiento y otra en
  validación/test. Partición estratificada por imagen 36/12/12 (60/20/20).
* K-Means: k=2, k-means++, 10 reinicios, máx. 300 iteraciones, tol=1e-4, ajuste sobre
  12 000 píxeles muestreados y asignación de todos los píxeles. Combinación elegida por
  Jaccard medio en desarrollo (entrenamiento+validación).
* Selección de características, SFS y PCA: solo entrenamiento. Umbral de la razón de
  verosimilitudes: índice de Youden en validación. El test no interviene en ninguna decisión.
* Análisis complementarios declarados (evalúan test, pero no modifican el sistema): IC
  bootstrap, ablación controlada de la máscara (misma selección, modelo y umbral), ablación
  de la media aritmética del tono y 30 particiones repetidas que repiten todo el protocolo,
  incluida la elección de la segmentación.
* K-Means se ejecuta con un hilo OpenMP para que la segmentación no dependa del número de
  núcleos de la máquina.

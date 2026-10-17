# Informe de la sesión — Clasificación de género musical con AST

Fecha: 2026-10-13 · Repositorio: `gxnzalx55/Music-Genre-Transformer-Sandbox` · Hardware: RTX 5070 (12 GB), Windows 11

## 1. Resumen

El proyecto partía de un **esqueleto**: toda la fase de datos estaba implementada, pero modelos, entrenamiento, evaluación y los tres scripts principales eran funciones vacías (`raise NotImplementedError`). En esta sesión se implementó todo lo que faltaba, se preparó el dataset FMA small y se entrenaron y compararon dos modelos.

| Modelo | Mejor val. acc | Test acc | Test macro-F1 | Tiempo |
|---|---|---|---|---|
| AST desde cero (9,67 M parám., 50 épocas) | 53,5 % (época 46) | **49,6 %** | 0,485 | ~11 min |
| AST preentrenado HF (15 épocas) | 63,3 % (época 11) | 56,75 % | 0,557 | ~38 min |
| AST preentrenado HF, `input_scale=0.5` (15 épocas) | **64,75 %** (época 8) | **58,63 %** | 0,578 | ~39 min |

El preentrenado se evaluó después en test (`outputs/pretrained_15ep/`): 56,75 % de precisión y 0,557 de macro-F1, unos 7 puntos por encima del modelo desde cero (49,6 %). Cae ~6,5 puntos respecto a validación (63,3 %), como se esperaba. Ambos quedan por debajo del rango que anuncia el README (55–65 % y 75–85 %); el desde cero (49,6 %) queda por debajo de 55–65 % y el preentrenado (56,75 %) muy lejos de 75–85 %.

## 2. Estado del código al empezar

Solo funcionaban `data/download.py`, `preprocess.py`, `stats.py`, `dataset.py` y los YAML. Los tests mostraban "5 passed, 5 skipped", pero los omitidos eran precisamente los de modelo y entrenamiento. El README presentaba el proyecto como funcional, y no lo era.

## 3. Trabajo realizado

Se hizo por bloques, con commit y push tras cada uno.

| Bloque | Contenido |
|---|---|
| 0 | Scripts: se añade la raíz del proyecto a `sys.path` (todos fallaban con `No module named 'data'`). |
| 1 | `training/utils.py` (dispositivo, semillas, `load_config`, contar parámetros), `scheduler.py` (warmup lineal + coseno por paso), `early_stopping.py`, `data/transforms.py` (SpecAugment por muestra en PyTorch puro, mixup). |
| 2 | `models/ast_scratch.py` (patch embed, DropPath, bloque con `scaled_dot_product_attention`, AST completo), `ast_pretrained.py` (wrapper HuggingFace con congelado de capas y LR diferenciado), `factory.py`. |
| 3 | `training/trainer.py`: autocast bf16, acumulación de gradientes, clipping, SpecAugment y mixup, TensorBoard/W&B opcional, early stopping, checkpoints top-k y `best.pt`. |
| 4 | `evaluation/metrics.py` y `report.py`, y `train.py`, `evaluate.py`, `predict.py` operativos. `load_model_from_checkpoint`. |

Tests: de 5 OK + 5 omitidos a **11 OK + 2 omitidos** (los omitidos son el conteo de parámetros del modelo desde cero y el forward del preentrenado).

Commits: `87531b3`, `ff665d2`, `1e203c0`, `34f539d` y el commit final de esta sesión.

## 4. Problemas encontrados y cómo se resolvieron

1. **Scripts sin `sys.path`.** Los comandos del README fallaban. Corregido en los 4 scripts.
2. **`checksums.txt` devuelve 404** en el mirror de FMA y abortaba `prepare_data.py` tras bajar 8 GB. Ahora su ausencia no es fatal.
3. **Los 8000 tracks marcados como corruptos.** Faltaban `torchaudio`, `librosa` y otras dependencias de `requirements.txt`; `is_corrupt()` se tragaba el `ImportError` y dejaba una caché vacía sin avisar. Se instalaron las dependencias y se añadió una comprobación que falla con un mensaje claro.
4. **PyTorch sin CUDA.** El venv tenía `torch 2.14.1+cpu`. Se reinstaló `torch`/`torchaudio` 2.11.0+cu128 (versión máxima disponible en el índice cu128); la GPU funciona.
5. **`transformers` 5.x** cambió la estructura interna del AST (`layers` frente a `encoder.layer`). El código admite ambas.
6. **Ventana duplicada** en `sliding_windows` cuando el clip acababa justo en el borde de una ventana (detectado por un test nuevo). No afectaba a clips de 30 s, pero se corrigió.
7. **`torch.compile` falla en Windows** (`TritonMissing`). `compile` pasa a `false` en `configs/base.yaml`.
8. **Memoria con el modelo preentrenado.** Con lote 32 el pico era 15 GB (> 12 GB de VRAM) y cada paso tardaba 36 s por el desbordamiento a memoria compartida; el primer intento perdió ~14 minutos antes de detectarlo y pararlo. Se midió y se usó lote 16 (pico 8,1 GB, ~150 s/época).

Un fallo mío también quedó registrado: dos veces di por buena la preparación de datos sin comprobar el código de salida real, y en ambas había fallado. Desde entonces el `SCRIPT_EXIT` se escribe en el log y se comprueba.

## 5. Resultados detallados

### AST desde cero (test, ventanas deslizantes de 10 s)

| Género | Precisión | Recall | F1 |
|---|---|---|---|
| Electronic | 0,559 | 0,660 | 0,606 |
| Experimental | 0,407 | 0,220 | 0,286 |
| Folk | 0,388 | 0,450 | 0,417 |
| Hip-Hop | 0,641 | 0,820 | 0,719 |
| Instrumental | 0,398 | 0,510 | 0,447 |
| International | 0,529 | 0,640 | 0,579 |
| Pop | 0,284 | 0,230 | 0,254 |
| Rock | 0,815 | 0,440 | 0,571 |

Precisión de entrenamiento ≈ 50 %, casi igual a la de validación: el modelo **se queda corto**, no sobreajusta. El early stopping nunca actuó. Hip-Hop, Electronic e International van mejor; Pop y Experimental peor.

### AST preentrenado (validación)

La validación llegó a 54 % en la **primera** época (el modelo desde cero tardó 46 en alcanzarlo) y se estabilizó en 62–63 % desde la época 7. La precisión de entrenamiento siguió subiendo hasta 72,7 % (época 15): empieza un sobreajuste leve. Mejor época: 11 (val. acc 0,6325, val. F1 0,6276).

Con `input_scale=0.5` (el AST original esperaba entradas de desviación ≈ 0,5) el test mejora 1,9 puntos; está dentro del ruido (±1,7) y es una sola ejecución. Ahora es el valor por defecto. El mejor punto fue la época 8; después la validación se estanca y el entrenamiento sobreajusta.

## 6. Limitaciones y cosas por verificar

- La validación usa **un recorte aleatorio** por clip (no ventanas), por lo que tiene algo de ruido.
- El AST preentrenado espera su propia normalización de entrada; aquí se normaliza con la media/desviación de FMA (−9,617 / 17,340). Puede restar rendimiento.
- Con `n_fft=400` y 128 mels, `torchaudio` avisa de que algún filtro mel queda vacío (decisión de la configuración).
- `torchaudio.load` no lee MP3 (falta `torchcodec`); el código cae a `librosa`, que sí funciona.
- El AST desde cero tiene 9,67 M parámetros, no los 5–8 M que indicaba el docstring del test, que por eso sigue omitido.
- **`torch.compile`, W&B y el resto de hiperparámetros no se han explorado.**

## 7. Siguientes pasos recomendados

1. ~~Evaluar el preentrenado en test~~ (hecho: 56,75 %).
2. ~~Corregir el README~~ (hecho).
3. ~~`best.pt` por experimento~~ (hecho: ahora se guarda en `checkpoints/<experimento>/best.pt`).
4. Probar menos regularización (mixup/SpecAugment) o más épocas en el modelo desde cero, y una normalización acorde al preentrenado.

## 8. Dónde está cada cosa

- Código y configs: en git (`main`).
- Checkpoints (`checkpoints/`), logs de TensorBoard (`logs/`), resultados (`outputs/scratch_50ep/`) y datos (`data/`): **no están en git** (ignorados). Los logs de texto `train_scratch.log`, `train_pretrained.log` y `prepare_data.log` están en la raíz sin versionar.
- Dependencias: `torch 2.11.0+cu128`, `transformers 5.19`.

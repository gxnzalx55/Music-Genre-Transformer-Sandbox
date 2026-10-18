# Informe — Etiquetado de playlists e integración de géneros propios en el dataset

Fecha: 2026-10-18 · Rama: `feature/ingesta_de_datos` · Continúa: `informes/02_Pipeline_dataset_YouTube.md` · Hardware: RTX 5070 (12 GB), Windows 11

## 1. Resumen

La ingesta del informe 02 producía segmentos log-mel pero sin etiqueta. El objetivo de esta sesión fue poder **añadir clases nuevas** al modelo (más allá de los 8 géneros de FMA) a partir de playlists públicas de Spotify, y controlar la calidad de las etiquetas, dado que el entrenamiento es supervisado. Solo se trabajó con Spotify; YouTube se mantiene en el código pero no se tocó ni se probó de nuevo.

Se implementó:

1. **Etiquetado por playlist**: una playlist equivale a un género (`--genre`, o varias a la vez con `--playlist GENERO=URL`). La etiqueta y la URL de origen quedan en `manifest.csv`.
2. **Control de calidad por audio** (`scripts/check_labels.py`): marca, por género, las canciones cuyo embedding de AST está lejos del centro de su clase.
3. **Integración en el dataset**: bloque opcional `dataset.custom` que añade las canciones propias a FMA y amplía `num_classes` automáticamente.
4. **Documentación** del flujo en el README (sección 5).

Estado: las piezas están cubiertas por tests unitarios (24 pasados, 2 omitidos), pero **no se ha ingerido ninguna playlist nueva ni se ha entrenado con clases propias**. Las únicas canciones reales son las 5 de AC/DC de la sesión anterior, sin género.

## 2. Archivos creados y modificados

| Archivo | Cambio |
|---|---|
| `scripts/ingest_playlist.py` | `--genre`, `--playlist GENERO=URL` (repetible); columnas `genre` y `playlist` en el manifiesto; migración de manifiestos antiguos; aviso si una pista ya existe con otro género; `main` separado en `parse_jobs` e `ingest_playlist` |
| `data/ingest/labeling.py` (nuevo) | `flag_outliers`: distancia coseno al centroide y z-score robusto (mediana/MAD) |
| `scripts/check_labels.py` (nuevo) | Embeddings con el AST ajustado, escribe `review.csv` |
| `data/custom.py` (nuevo) | Carga de canciones propias, split por hash, ampliación de géneros |
| `data/dataset.py` | `FMADataset` usa una lista plana de muestras (FMA + propias) |
| `training/utils.py` | `load_config` amplía `genres`/`num_classes` si `custom` está activo |
| `scripts/evaluate.py` | Evalúa con las clases guardadas en el checkpoint |
| `configs/dataset/fma_small.yaml` | Bloque `custom` (desactivado por defecto) |
| `tests/test_ingest.py`, `tests/test_custom.py` | Tests nuevos (migración, outliers, mezcla, splits, exclusiones) |
| `README.md` | Sección «Add your own genres (Spotify playlists)» y árbol del proyecto |

## 3. Diseño

**Origen de la etiqueta.** Se descartó usar los géneros de Spotify: `spotdl save` devuelve el campo `genres` vacío (`['', '']`), comprobado con Thunderstruck. Tampoco sirven las pseudo-etiquetas del propio modelo, porque no conoce las clases nuevas. La etiqueta la da el usuario por playlist, lo que asume que cada playlist es de un solo género.

**Control de calidad.** Para cada género con al menos `--min-tracks` canciones (10 por defecto):
- Embedding por canción: media sobre ventanas deslizantes de la entrada del clasificador del AST (salida agrupada, antes de `LayerNorm` y cabeza), con el checkpoint `pretrained_scale05`.
- Centroide = mediana de los embeddings normalizados; distancia = 1 − coseno.
- z-score robusto sobre las distancias; `outlier` si z > 3 (`--z`).
- Escribe `review.csv` ordenado de más a menos sospechoso. **Solo marca, no borra.**

**Integración en el dataset** (`dataset.custom.enabled=true`):
- Los géneros del manifiesto se unen a los 8 de FMA. Un género con el mismo nombre que uno de FMA se fusiona con esa clase (distingue mayúsculas).
- Split 80/10/10 de las canciones propias por hash SHA-1 del `track_id`: estable, de modo que una canción no pasa de test a entrenamiento al añadir más. FMA conserva su split oficial.
- Se excluyen canciones sin género, las marcadas `outlier` (`exclude_outliers`) y géneros con menos de `min_tracks` canciones (30 por defecto, con aviso en el log).
- Desactivado por defecto: sin la opción, el comportamiento es idéntico al anterior (comprobado: 8 clases, 6394 muestras de entrenamiento).

## 4. Problemas encontrados y cómo se resolvieron

| Problema | Resolución |
|---|---|
| `genres` de spotDL vacío | La etiqueta pasa a ser por playlist; el control de calidad se hace con audio |
| El manifiesto antiguo no tenía `genre` | Migración automática al abrir el `Store`; las 5 pistas de AC/DC quedan con género vacío |
| `evaluate.py` leía las clases del YAML, no del checkpoint | Ahora usa las clases del checkpoint, evitando desajustes si el manifiesto cambia tras entrenar |
| Con 5 pistas, una de AC/DC (Thunderstruck) salía `outlier` siendo claramente rock | Esperado con una muestra tan pequeña; de ahí el umbral `--min-tracks` |
| Error de sintaxis al reescribir `main` (saltos de línea dentro de cadenas) | Corregido a mano; tests y casos de error del CLI verificados después |
| Rama remota ya existente con commits propios | Ver sección 6 |

## 5. Verificación realizada

- `pytest`: **24 pasados, 2 omitidos** (también tras aplicar los commits sobre la rama remota).
- Tests nuevos: migración del manifiesto y guardado del género; `flag_outliers` detecta una canción fuera de clase; split estable y proporcional (±3 %); ampliación de géneros y fusión con clases de FMA; exclusión de outliers y géneros pequeños; mezcla FMA + propias por split sin solapes; con `custom` desactivado el dataset no cambia.
- CLI de ingesta: los casos de error (sin URL, `--playlist` mal formado, falta de género) devuelven el mensaje esperado.
- `check_labels.py`: ejecutado de punta a punta con una copia temporal de las 5 pistas etiquetadas como «Rock»: genera `review.csv` correctamente. Sin género en el manifiesto real, termina con un mensaje claro.
- Config real: `load_config` con `custom` desactivado y activado da 8 clases (no hay canciones propias etiquetadas).

## 6. Control de versiones

Se crearon tres commits en `feature/ingesta_de_datos`: ingesta con etiquetas y control de calidad; integración en el dataset; documentación e informes. Al hacer push, la rama ya existía en el remoto con dos commits previos (`docs/` con la guía del Transformer y un cambio en la demo de Gradio) y con el historial antiguo reescrito (mismos mensajes, otros hashes, sin ancestro común). Los tres commits se aplicaron limpios encima de esa rama, se volvieron a pasar los tests y el push fue un avance normal, sin forzar. La copia original local quedó en `backup/ingesta-local-original`.

## 7. Limitaciones y notas

- **La etiqueta es tan buena como la playlist.** Una playlist mixta mete ruido en su clase; el control de calidad lo reduce pero no lo elimina.
- **No probado de punta a punta**: ni ingesta de playlists reales con género, ni `check_labels.py` sobre una clase de tamaño real, ni `train.py`/`evaluate.py` con clases propias.
- **Desbalance**: FMA small tiene unas 1000 canciones por clase. Una clase nueva con decenas de canciones se aprende mal, y con 30 canciones el 10 % da unas 3 en validación y 3 en test, con métricas muy ruidosas. Convendría apuntar a cientos de canciones por género o ponderar la pérdida.
- **Hay que reentrenar**: los tres checkpoints actuales tienen 8 salidas.
- **Etiqueta fija por canción**: una pista ya ingerida no se reetiqueta al volver a ingerirla con otro género; hay que editar `manifest.csv`.
- **Audio de YouTube Music**: la comprobación de duración (5 s) reduce, sin eliminar, el riesgo de una versión en directo o un remix.
- **Las 5 pistas de AC/DC** siguen sin género y quedan fuera del dataset hasta etiquetarlas.
- Los splits por hash no están estratificados: con géneros pequeños algún split puede quedar casi vacío (de ahí `min_tracks`).

## 8. Siguientes pasos

1. Ingerir 2–3 playlists reales con género y ejecutar `check_labels.py` para ver el comportamiento con tamaños realistas.
2. Entrenamiento corto con `dataset.custom.enabled=true` para validar el flujo completo (entrenar → evaluar → `predict`).
3. Valorar pérdida ponderada o muestreo balanceado para las clases pequeñas.
4. Etiquetar o descartar las 5 pistas de AC/DC.
5. Probar las pendientes de Spotify del informe 02: Ctrl+C a mitad y pista sin coincidencia.

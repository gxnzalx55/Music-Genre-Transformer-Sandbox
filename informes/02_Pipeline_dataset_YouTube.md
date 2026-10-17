# Informe — Pipeline de dataset propio desde YouTube y Spotify

Fecha: 2026-10-16 · Rama: `develop` · Especificación: `informes/Pipeline de dataset propio desde YouTube.md` · Hardware: RTX 5070 (12 GB), Windows 11

## 1. Resumen

Se implementó el script `scripts/ingest_playlist.py`, que recibe la URL de una playlist de YouTube y genera, por cada canción, un segmento de 30 s que empieza al 35 % de su duración, guardado como log-mel `.npy` con los mismos parámetros que la caché de FMA. El audio se descarga solo en el tramo necesario y no se recomprime a MP3.

```
python scripts/ingest_playlist.py <URL_PLAYLIST> --out data/custom
```

Además admite playlists de **Spotify** (sección 9): la CLI detecta el origen por la URL.

Estado (YouTube): la parte de procesado de una canción está verificada con un vídeo real de YouTube. **La CLI completa sobre una playlist todavía no se ha ejecutado**, porque el paso 1 (crear la playlist) es manual. Ver sección 6.

## 2. Archivos creados y modificados

| Archivo | Tipo | Contenido |
|---|---|---|
| `scripts/ingest_playlist.py` | nuevo | CLI (`--out`, `--config`, `--limit`, `--dry-run`, `--workers`), clase `Store` (estado en disco), bucle con progreso y resumen final |
| `data/ingest/spotify.py` | nuevo | Listado de playlists de Spotify y descarga de la canción completa con spotDL (sección 9) |
| `data/ingest/youtube.py` | nuevo | Listado plano de la playlist, metadatos individuales, descarga parcial con yt-dlp |
| `data/ingest/segment.py` | nuevo | `compute_segment`, localización de ffmpeg, decodificación a 16 kHz mono, `audio_to_mel` |
| `configs/ingest/default.yaml` | nuevo | `offset_ratio`, `segment_seconds`, `min_duration`, pausas, reintentos, config de mel a reutilizar |
| `tests/test_ingest.py` | nuevo | 6 tests (cálculo del tramo y paridad del mel con FMA) |
| `data/preprocess.py` | modificado | Se extrae `waveform_to_melspec` de `compute_melspec`; sin cambio de comportamiento |
| `scripts/predict.py` | modificado | Acepta un `.npy` además de audio |
| `requirements.txt` | modificado | Añade `yt-dlp` e `imageio-ffmpeg` |
| `.gitignore` | modificado | Ignora `data/custom/` |
| `README.md` | modificado | Enlaces a la carpeta `informes/` |

Estructura de salida generada por el script:

```
data/custom/
  mels/<video_id>.npy     un segmento por canción
  manifest.csv            video_id, title, channel, duration, start, end, path
  archive.txt             vídeos ya procesados (reanudación)
  failed.log              video_id <tab> motivo
```

## 3. Diseño

**Flujo por canción** (`process_one`, nunca lanza excepciones):

1. Comprobar si la entrada es `[Private video]` o `[Deleted video]` → fallo registrado.
2. Si el listado plano no trae `duration`, pedir los metadatos del vídeo.
3. `compute_segment`: `inicio = duración × 0,35`, `fin = inicio + 30`. Si `fin` supera la duración, la ventana se desplaza hacia atrás; si dura menos de 30 s, se descarta.
4. Descargar solo `[inicio, fin]` con `bestaudio`, `download_ranges` y `force_keyframes_at_cuts`, a un temporal, con 3 reintentos.
5. Decodificar con ffmpeg a 16 kHz mono float32, recortar o rellenar a exactamente 30 s y calcular el log-mel.
6. Escribir `mels/<id>.npy.tmp`, renombrar a `.npy`, añadir la fila al manifiesto y la línea a `archive.txt`.
7. Borrar el temporal y esperar una pausa aleatoria (2–5 s).

**Paridad con FMA.** El cálculo del mel se extrajo a `waveform_to_melspec`, que usan tanto `compute_melspec` (FMA) como el nuevo pipeline. Los parámetros se leen de `configs/dataset/fma_small.yaml` (128 bandas, n_fft 400, hop 160, `top_db` 80), por lo que no puede haber divergencia de configuración.

**Reanudación y robustez.** Un vídeo cuenta como hecho si está en `archive.txt` o si aparece en `manifest.csv` con su `.npy` existente. La segunda condición cubre un corte entre la escritura del manifiesto y la del archivo. Las escrituras de manifiesto y archivo van bajo un `Lock`, lo que permite `--workers N`. Ante Ctrl+C se cancelan las tareas pendientes, se limpia `tmp/` y se indica que basta relanzar el mismo comando.

**Nombres por `video_id`**, nunca por título (los títulos se repiten y contienen caracteres no válidos en Windows).

## 4. Problemas encontrados y cómo se resolvieron

1. **yt-dlp no encuentra ffmpeg.** El PC no tiene ffmpeg en el PATH; el proyecto `Yt-mp3` usa el binario de `imageio-ffmpeg`. Hubo dos causas encadenadas:
   - yt-dlp solo reconoce un ejecutable llamado literalmente `ffmpeg[.exe]`, y el de imageio se llama `ffmpeg-win-x86_64-v7.1.exe`.
   - Su comprobación de descarga parcial (`FFmpegFD.available`) ignora la opción `ffmpeg_location` y solo mira el PATH (el propio código lo marca con un `TODO`/`Fixme`).

   Solución en `get_ffmpeg()`: si no hay ffmpeg en el PATH, se copia el binario de imageio como `ffmpeg.exe` en `%TEMP%\mgc_ffmpeg` y se añade esa carpeta al PATH del proceso. Si ya hay un ffmpeg en el PATH, se usa ese.
2. **Heredoc de shell fallido** al crear los archivos en bloque: no se creó nada y se rehízo con la herramienta de escritura. Sin impacto en el resultado.

## 5. Verificación realizada

**Tests.** `pytest tests/test_ingest.py`: 6 passed. Suite completa: **17 passed, 2 skipped** (los 2 omitidos ya lo estaban antes, ver informe 01).

| Test | Comprueba |
|---|---|
| `test_normal_song_starts_at_35_percent` | 200 s → 70–100 s |
| `test_short_song_is_discarded` | 29,9 s y duración `None` → descartada |
| `test_end_beyond_duration_shifts_back` | 40 s → 10–40 s |
| `test_exactly_30s_song` | 30 s → 0–30 s |
| `test_custom_ratio_and_length` | parámetros no por defecto |
| `test_audio_to_mel_matches_fma_pipeline` | mismo WAV por `compute_melspec` y por `audio_to_mel`: forma (128, 3001), float32, `allclose` a 1e-3 |

**Prueba real con YouTube** (un vídeo público de 213 s, más una entrada `[Private video]`):

| Comprobación | Resultado |
|---|---|
| Duración ausente del listado → metadatos individuales | OK |
| Tramo calculado | 74,55 s – 104,55 s (35 % de 213 s) |
| `.npy` generado | forma (128, 3001), float32 |
| Escala frente a la caché FMA | yt: mín −46,9 / máx 33,1 / media 2,4 · FMA: mín −47,7 / máx 32,3 / media 1,0 |
| Fila en `manifest.csv` | correcta |
| Vídeo privado | queda en `failed.log` (`unavailable ([Private video])`) y no detiene el proceso |
| Carpeta `tmp/` tras la ejecución | vacía |

**`predict.py` con `.npy`** y el checkpoint `pretrained_scale05_AST-HuggingFace/best.pt`: Pop 0,444 · International 0,212 · Electronic 0,149. El vídeo es una canción pop, por lo que el top-1 es coherente; es una sola muestra elegida por mí, no una evaluación.

## 6. Criterios de aceptación

| Criterio | Estado |
|---|---|
| `--limit 5` genera 5 `.npy` y 5 filas en el manifiesto | **Pendiente** de probar con una playlist real (la lógica por canción sí está probada) |
| Relanzar sobre la misma playlist no descarga nada | **Pendiente** (implementado vía `archive.txt` y manifiesto; sin probar de extremo a extremo) |
| Ctrl+C y relanzar continúa sin archivos corruptos | **Pendiente** de probar (escritura atómica `.tmp` + `os.replace` implementada) |
| Vídeo privado o eliminado en `failed.log` sin detener el proceso | Verificado con una entrada `[Private video]`; falta con un fallo real de la API |
| `.npy` con misma forma y escala que uno de FMA | Verificado (forma y rango; ver diferencia de longitud abajo) |
| `predict.py` lee `.npy` y devuelve un top-3 coherente | Verificado con una muestra; ver sección 7 |
| Tests de pytest del cálculo del tramo | Cumplido |

## 7. Limitaciones y notas

- **Longitud 3001 frente a ~2998.** Los clips de FMA duran en realidad unos 29,98 s (2998 frames). No afecta al modelo porque se trocea en ventanas de 10 s.
- **Desviación respecto a la especificación:** se usa un `archive.txt` propio en lugar de `download_archive` de yt-dlp, para que un vídeo solo conste como hecho cuando el `.npy` y la fila del manifiesto ya están escritos.
- **Vídeos con restricción de edad:** sin cookies yt-dlp los rechaza y acaban en `failed.log`. No hay soporte de cookies.
- **Parche del PATH:** `get_ffmpeg()` modifica `os.environ["PATH"]` del proceso. Es una solución a una limitación de yt-dlp y puede eliminarse si una versión futura respeta `ffmpeg_location`.
- **Canciones cortas** se registran en `failed.log` como "too short" y no se archivan, así que se reevalúan en cada ejecución (coste mínimo, el dato viene del listado).
- Un solo vídeo de prueba no valida la calidad del segmento a escala: no se ha comprobado qué fracción de los tramos al 35 % cae en intros, silencios o partes no representativas.
- No se ha hecho commit de este trabajo.

## 8. Siguientes pasos

1. Crear la playlist y ejecutar `--dry-run`, luego `--limit 5`, relanzar y probar Ctrl+C para cerrar los criterios pendientes.
2. Definir las secciones TODO de la especificación: base de datos, etiquetado, filtros e integración con el sandbox.
3. Decidir cómo se etiquetarán los géneros de las canciones propias respecto a las 8 clases de FMA.

## 9. Soporte de playlists de Spotify

```
.venv\Scripts\python.exe scripts/ingest_playlist.py <URL_SPOTIFY> --out data/custom
```

La salida es idéntica a la de YouTube (mismo `mels/`, `manifest.csv`, `archive.txt` y `failed.log`); en el manifiesto `source=spotify` y `track_id` es el ID de pista de Spotify.

**Diseño.**
- `is_spotify_url` decide el origen (`open.spotify.com` o `spotify:`).
- `list_playlist` ejecuta `spotdl save` y devuelve id, título, artistas, duración y URL de cada pista.
- Spotify no permite descargar un tramo suelto. `download_full` baja la canción completa con spotDL (la empareja en YouTube Music) en m4a.
- Se compara la duración real con la de Spotify (`duration_tolerance`, 5 s). Si difiere, se descarta como "wrong match" para evitar versiones en directo, remixes o vídeos incorrectos.
- Se recorta `[35 % de la duración, +30 s]` con ffmpeg a 16 kHz mono y se calcula el log-mel con los mismos parámetros que FMA. La canción completa se borra siempre al terminar.
- spotDL vive en un entorno aparte (`.venv-spotdl`) para no mezclar dependencias; su ruta está en `configs/ingest/default.yaml` (`spotdl_exe`).

**Prueba real** con la playlist indicada por el usuario, con `--limit 5` (5 pistas de AC/DC):

| Comprobación | Resultado |
|---|---|
| Procesadas / fallidas | 5 / 0, en 206 s (~36 s por pista) |
| Filas en `manifest.csv` | 5, con `source=spotify` y tramos al 35 % (p. ej. Thunderstruck, 292 s → 102,2–132,2 s) |
| `.npy` generados | forma (128, 3001), float32, rango aprox. −44 a 39 (equivalente a FMA) |
| Relanzar el mismo comando | 0 procesadas, 5 ya existentes, sin descargar nada |
| `predict.py` sobre `.npy` (checkpoint `pretrained_scale05`) | Thunderstruck: Rock 0,919 · International 0,022 · Pop 0,020. Highway to Hell: Rock 0,912 · Pop 0,032 · Hip-Hop 0,020. Coherente con el género |

**Limitaciones.**
- Más lento que YouTube (~36 s por pista) porque se descarga la canción entera.
- El audio no sale de Spotify sino del emparejamiento de spotDL en YouTube; la comprobación de duración reduce, sin eliminar, el riesgo de una versión equivocada.
- Pendientes de probar con Spotify: Ctrl+C a mitad, y una pista sin coincidencia (debería quedar en `failed.log`). La playlist completa tampoco se ha procesado; solo las 5 primeras pistas.
- Los criterios pendientes de la sección 6 quedan cubiertos para Spotify en lo referente a `--limit 5` y relanzar sin descargar; siguen pendientes para YouTube.

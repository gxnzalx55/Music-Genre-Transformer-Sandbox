# Pipeline de dataset propio desde YouTube

Oct 16, 2026 · @MiLord

## Resumen

Un único script recibe el enlace de una playlist de YouTube y genera, por cada canción, un segmento de 30 s extraído a partir del 35 % de su duración y guardado como log-mel `.npy`, listo para el sandbox. El audio nunca se recomprime a MP3: se descarga solo el tramo necesario y se convierte directamente a 16 kHz mono.

Uso final previsto:

```
python scripts/ingest_playlist.py <URL_PLAYLIST> --out data/custom
```

Base de datos, etiquetado, filtros e integración quedan como pendientes al final de este documento.

## Requisitos y estructura

El script vive dentro de `Music-Genre-Transformer-Sandbox` y reutiliza su configuración de mel, para que los `.npy` sean idénticos a los de FMA.

| Componente | Uso |
| --- | --- |
| `yt-dlp` (como librería) | Listar la playlist y descargar solo el tramo de 30 s |
| `ffmpeg` | Recorte y conversión a 16 kHz mono (ya incluido en tu repo Yt-mp3) |
| `torchaudio` / `librosa` | Cálculo del log-mel con los parámetros del sandbox |
| `tqdm` | Progreso por canción |

Archivos nuevos:

```
scripts/ingest_playlist.py      CLI de entrada
data/ingest/youtube.py          listado y descarga con yt-dlp
data/ingest/segment.py          cálculo del tramo y log-mel
configs/ingest/default.yaml     parámetros (offset, duración, pausas)
data/custom/
  mels/<video_id>.npy           un segmento por canción
  manifest.csv                  video_id, título, canal, duración, inicio, fin, ruta
  archive.txt                   vídeos ya procesados (reanudación)
  failed.log                    errores con motivo
```

Nombra los archivos por `video_id`, nunca por el título: los títulos se repiten y contienen caracteres no válidos en Windows.

## Pasos de implementación

Cada paso se implementa y se prueba antes de pasar al siguiente.

1. **Crear la playlist (manual).** Una playlist pública o no listada en YouTube. Copia su URL (`https://www.youtube.com/playlist?list=...`).
2. **Listar la playlist sin descargar.** `yt_dlp` con `extract_flat: "in_playlist"` devuelve `id`, `title`, `channel` y `duration` de cada vídeo. Nada de scraping HTML.
   - Si `duration` falta en el listado plano, pedir los metadatos del vídeo individual antes de descargar.
   - Saltar vídeos privados, eliminados o con restricción de edad, y anotarlos en `failed.log`.
3. **Calcular el tramo.** `inicio = duración × 0,35` y `fin = inicio + 30 s`.
   - Ambos valores configurables (`offset_ratio`, `segment_seconds`).
   - Si `fin` supera la duración, desplazar el tramo hacia atrás para que siga midiendo 30 s.
   - Si la canción dura menos de 30 s, descartarla.
4. **Descargar solo ese tramo.** `yt_dlp` con `format: "bestaudio"`, `download_ranges` sobre `[inicio, fin]` y `force_keyframes_at_cuts: True`, a un archivo temporal.
   - Sin postprocesador MP3: el audio queda en su formato original (opus o m4a).
   - Activar `download_archive` con `archive.txt` para reanudar sin repetir.
   - Pausas entre descargas (`sleep_interval` 2–5 s) y reintentos (`retries: 3`).
5. **Convertir y calcular el log-mel.** Decodificar el temporal a 16 kHz mono y calcular el log-mel con los mismos parámetros que `configs/dataset/fma_small.yaml` (128 bandas, ventana 25 ms, hop 10 ms).
   - Comprobar que el array resultante tiene la forma esperada (128 × \~3000 frames).
6. **Guardar y limpiar.** Escribir `mels/<video_id>.npy`, añadir una fila a `manifest.csv` y borrar el temporal.
   - Escribir primero a un archivo `.tmp` y renombrar: un corte a mitad no deja archivos corruptos.
7. **Resumen final.** Al terminar, imprimir procesados, saltados ya existentes, fallidos y tiempo total.

Opciones de la CLI: `--limit N` para probar con pocas canciones, `--dry-run` para listar sin descargar y `--workers N` para descargas en paralelo (por defecto 1, para no forzar a YouTube).

## Criterios de aceptación

El script está terminado cuando cumple todo lo siguiente:

- [ ] Con `--limit 5` genera 5 `.npy` y 5 filas en `manifest.csv`.
- [ ] Relanzarlo sobre la misma playlist no vuelve a descargar nada.
- [ ] Interrumpirlo con Ctrl+C y relanzarlo continúa donde se quedó, sin archivos corruptos.
- [ ] Un vídeo privado o eliminado queda en `failed.log` y no detiene el proceso.
- [ ] Un `.npy` generado tiene la misma forma y escala que uno de la caché de FMA.
- [ ] `predict.py` adaptado para leer un .npy devuelve, con el checkpoint de FMA, un top-3 coherente con el género de la canción (hoy espera un MP3).
- [ ] Tests con pytest para el cálculo del tramo (canción normal, canción corta, fin fuera de rango).

## Pendiente (TODO)

Se definirán en conversaciones siguientes; aquí queda lo ya acordado como punto de partida.

### Base de datos

### Etiquetado

### Filtros

### Integración con el sandbox

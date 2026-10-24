# Informe — Ingesta de la playlist de Metal: bloqueo en spotDL y retomada pendiente

Fecha: 2026-10-20 · Rama: `feature/ingesta_de_datos` · Continúa: `informes/03_Etiquetado_e_integracion_de_generos_propios.md` · Hardware: RTX 5070 (12 GB), Windows 11

## 1. Resumen

El objetivo era el paso 1 del informe 03: ingerir una playlist real con género (Metal, 712 canciones) y pasar `check_labels.py`. **No se completó**: la ingesta se quedó bloqueada en el listado de la playlist, sin procesar ninguna pista, y se canceló a mano. No se ha añadido ningún dato nuevo; el estado de `data/custom` es el mismo que antes de la sesión (5 pistas de AC/DC sin género).

La ingesta se deja para más tarde. Este informe recoge lo ocurrido y cómo retomarlo.

## 2. Qué se hizo

- Playlist: `https://open.spotify.com/playlist/3kBjTaxe1p4mgNBPkU3HHo` (se quitó el parámetro `?si=`, que no hace falta), 712 canciones según el usuario.
- Comando previsto: `.venv\Scripts\python.exe scripts/ingest_playlist.py <URL> --genre Metal --out data/custom`.

## 3. Incidencias

| Problema | Detalle | Resolución |
|---|---|---|
| Primer intento falló al instante | Se lanzó con el Python del sistema, que no tiene `omegaconf` (`ModuleNotFoundError`) | Se relanzó con `.venv\Scripts\python.exe`, el entorno del proyecto |
| Segundo intento sin progreso | El proceso `spotdl save <playlist> --save-file ...\list.spotdl` llevaba ~18 min con 0 de CPU, sin conexiones de red abiertas, sin `ffmpeg`/`yt-dlp` y sin haber escrito `list.spotdl`. `mels/` y `manifest.csv` no cambiaron | Se mató el proceso `spotdl.exe` (PID 3472); la tarea en segundo plano terminó con error |
| Log vacío | La salida del script va con buffer y el log quedó en 0 bytes, así que no se pudo ver en qué punto estaba | Pendiente: ejecutar con `python -u` para ver la salida en directo |

**Causa probable (no confirmada):** límite de peticiones de la API de Spotify (error 429). spotDL puede quedarse esperando el `Retry-After`, que a veces es de muchos minutos. No hay evidencia directa porque el log estaba vacío; se deduce de que el bloqueo ocurre en el listado, sin red activa ni CPU.

## 4. Estado al cerrar

- `data/custom/mels/`: 5 ficheros (AC/DC, de la sesión anterior).
- `data/custom/manifest.csv`: cabecera + 5 filas, sin cambios; sin género.
- Ningún proceso de `spotdl`, `ffmpeg` ni `ingest_playlist` en ejecución.
- No se modificó código ni configuración.

## 5. Cómo retomarlo

1. Esperar un rato (30–60 min) para que se enfríe el posible límite de Spotify.
2. Lanzar desde la raíz del proyecto, con salida sin buffer y guardada en un log:
   ```powershell
   .venv\Scripts\python.exe -u scripts/ingest_playlist.py "https://open.spotify.com/playlist/3kBjTaxe1p4mgNBPkU3HHo" --genre Metal --out data/custom 2>&1 | Tee-Object ingest_metal.log
   ```
3. Si el listado vuelve a colgarse, probar `.venv-spotdl\Scripts\spotdl.exe save <URL> --save-file lista.spotdl` a mano para ver el error de spotDL en directo.
4. La ingesta es incremental (`archive.txt` y manifiesto): si se corta a mitad, relanzar el mismo comando continúa sin repetir lo ya hecho.
5. Para vigilar el avance en PowerShell:
   ```powershell
   while ($true) { Clear-Host; "mels: " + (Get-ChildItem data\custom\mels).Count; Start-Sleep 5 }
   ```
6. Al terminar: revisar cuántas pistas entraron y cuántas fallaron, ejecutar `scripts/check_labels.py` y revisar `review.csv`.

## 6. Valoración previa de la clase Metal

Con 712 canciones, el split 80/10/10 por hash daría unas 570 en train, 71 en val y 71 en test: suficiente para métricas razonables y un desbalance leve frente a las ~1000 de cada clase de FMA. La pérdida ponderada pasa a ser opcional. Metal es una clase aparte (no se fusiona con Rock de FMA) y es previsible cierta confusión con Rock.

## 7. Siguientes pasos

1. Retomar la ingesta (sección 5).
2. `check_labels.py` sobre Metal con tamaño real.
3. Entrenamiento corto con `dataset.custom.enabled=true` para validar el flujo completo (entrenar → evaluar → `predict`).
4. Etiquetar o descartar las 5 pistas de AC/DC.
5. Probar las pendientes de Spotify del informe 02: Ctrl+C a mitad y pista sin coincidencia.

## 8. Actualización (2026-10-24): causa confirmada y abandono de la vía Spotify

**Estado final: ingesta fallida y abandonada.** Rama marcada con el commit `91e6257 FAILED: Spotify playlist ingestion abandoned`.

### Diagnóstico
- Se ejecutó `spotdl save` a mano con `--ffmpeg` (el de `imageio-ffmpeg`) y `--log-level DEBUG`. El listado inicial funciona (`Found 712 songs` en ~7 s), pero después el proceso se queda parado (0 de CPU, sin red) durante más de 4 min sin escribir `list.spotdl`.
- Llamada directa a la API de Spotify con las credenciales de `~/.spotdl/config.json`: el token se obtiene bien, pero la petición a la API devuelve **HTTP 429 con `Retry-After: 86400`** (24 h).
- El `client_id` configurado (`5f573c96…`) es el público que spotDL trae por defecto y comparten todos sus usuarios. El límite se aplica a ese `client_id`, no a la IP local, así que esperar no garantiza que se resuelva.
- Esto confirma la hipótesis de la sección 3: spotipy duerme el `Retry-After` completo al pedir los metadatos de cada canción.

### Hallazgo en el código
`data/ingest/spotify.py` (`_run`) llama a `subprocess.run` sin `timeout`, por lo que un 429 largo cuelga la ingesta indefinidamente y sin mensaje. No se modificó.

### Solución descartada
Crear una app propia en https://developer.spotify.com/dashboard y poner su `client_id`/`client_secret` en `~/.spotdl/config.json`. Se decidió no seguir por esta vía.

### Estado
- Sin cambios en código, configuración ni datos. `data/custom` sigue con las 5 pistas de AC/DC sin género.
- Los pasos 1–3 de la sección 7 (ingesta de Metal, `check_labels.py`, entrenamiento con datos propios) quedan sin hacer. Habría que buscar otra fuente para la clase Metal.

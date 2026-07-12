# CanchaVision

Sistema de video-tracking deportivo para **fútbol 11** y **futsal**. Analiza vídeo
de una cámara elevada, detecta y sigue a todos los jugadores, los clasifica por
equipo y (por fases) genera estadísticas para clubes y entrenadores.

> Estado: **prototipo / validación de idea**. Procesamiento en diferido (sobre
> vídeo grabado), pensado para correr en CPU.

## Arquitectura del pipeline

```
vídeo ─► [detect] ─► [track] ─► [teams] ─► [annotate] ─► vídeo anotado
                        │                       │
                        └────► [pitch] ─► [stats] ─► métricas (fase 2-3)
```

| Módulo (`src/canchavision/`) | Responsabilidad |
|---|---|
| `detect.py`   | Detección de jugadores/balón con YOLO (Ultralytics) |
| `track.py`    | IDs consistentes entre frames con ByteTrack |
| `teams.py`    | Clasificación por equipo según color de camiseta |
| `annotate.py` | Dibuja cajas, IDs, colores de equipo y balón |
| `pitch.py`    | Homografía campo→metros (Fase 2) |
| `stats.py`    | Distancia, velocidad, posesión, heatmaps (Fase 3) |
| `pipeline.py` | Orquesta todo el flujo sobre un vídeo |
| `cli.py`      | Punto de entrada por línea de comandos |

## Roadmap

- **Fase 1 (esqueleto actual):** detección + tracking + equipos → vídeo anotado.
- **Fase 2:** homografía → vista cenital y heatmaps.
- **Fase 3:** estadísticas (distancia, velocidad, posesión por zonas).
- **Fase 4:** adaptación a futsal + dashboard para el entrenador.

## Requisitos

- Python 3.10+
- ~2-3 GB de descargas la primera vez (PyTorch CPU + modelos)
- CPU (funciona) o GPU NVIDIA (mucho más rápido, opcional)

## Puesta en marcha

```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell

# 2. Instalar en modo editable
pip install -e .

# 3. Colocar un clip en data/raw/ y ejecutarlo
canchavision --video data/raw/mi_partido.mp4 --max-frames 300
```

La primera ejecución descarga automáticamente el modelo `yolov8n.pt`. El vídeo
anotado se guarda en `outputs/`.

### Config

`config/default.yaml` (fútbol 11) y `config/futsal.yaml` (futsal). Se elige con
`--config`.

## Notas

- Los pesos COCO (`yolov8n.pt`) detectan `person` y `sports ball` sin entrenar
  nada. Para mejor precisión se cambia luego a un modelo específico de fútbol
  (p. ej. datasets de Roboflow / SoccerNet).
- La clasificación de equipos por color se fija en los primeros frames con
  suficientes jugadores y se mantiene estable el resto del clip.

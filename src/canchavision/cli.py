"""Interfaz de línea de comandos de CanchaVision."""
from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="canchavision",
        description="Detecta, sigue y anota jugadores en vídeo de fútbol/futsal.",
    )
    parser.add_argument("--video", required=True, help="Ruta al vídeo de entrada.")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Ruta al YAML de configuración (default: fútbol 11).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Ruta del vídeo anotado (default: outputs/<nombre>_annotated.mp4).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Procesar solo los primeros N frames (útil para pruebas rápidas).",
    )
    parser.add_argument(
        "--start-sec", type=float, default=0.0,
        help="Saltar a este segundo del vídeo antes de empezar (ej. 600 = min 10).",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Procesar 1 de cada N frames (2-3 = 2-3x más rápido, mínima pérdida "
             "de precisión). Las stats se ajustan al fps efectivo automáticamente.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(
        args.video, cfg, args.output, args.max_frames, args.start_sec, args.stride
    )


if __name__ == "__main__":
    main()

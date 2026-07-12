"""Descarga un clip de muestra para probar el pipeline.

Uso:
    python scripts/download_sample.py <URL_de_YouTube>

Requiere yt-dlp (pip install -e ".[dev]"). Guarda el clip en data/raw/.
Consejo: usa vídeos con cámara táctica/elevada (se ve casi todo el campo).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEST = Path("data/raw")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    url = sys.argv[1]
    DEST.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "yt-dlp",
                "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
                "-o", str(DEST / "%(title)s.%(ext)s"),
                "--merge-output-format", "mp4",
                url,
            ],
            check=True,
        )
    except FileNotFoundError:
        print("yt-dlp no está instalado. Ejecuta: pip install -e \".[dev]\"")
        sys.exit(1)


if __name__ == "__main__":
    main()

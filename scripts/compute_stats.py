"""Recalcula las stats físicas desde un archivo de posiciones (instantáneo).

Separa lo lento (detección/homografía) de lo rápido (estadística): permite
afinar suavizado, umbral de sprint o tope de velocidad sin reprocesar el vídeo.
Aplica además la fusión de rastros (identidad) salvo que se pase --no-merge.

Uso:
    python scripts/compute_stats.py outputs/<video>_positions.csv [fps] [--no-merge]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from canchavision.merge import apply_mapping, merge_tracks
from canchavision.stats import MetricStats


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    do_merge = "--no-merge" not in sys.argv
    pos = [a for a in sys.argv[2:] if not a.startswith("--")]
    fps = float(pos[0]) if pos else 25.0

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_before = len({int(r["track_id"]) for r in rows})
    if do_merge:
        mapping = merge_tracks(rows, fps=fps)
        rows = apply_mapping(rows, mapping)
        n_after = len(set(mapping.values()))
        print(f"Fusión de rastros: {n_before} IDs -> {n_after} jugadores")
    else:
        print(f"Sin fusión: {n_before} IDs")

    rows.sort(key=lambda d: (int(d["track_id"]), int(d["frame"])))
    metric = MetricStats(fps=fps)
    for d in rows:
        metric.tracks[int(d["track_id"])].append((float(d["x_m"]), float(d["y_m"])))

    metrics = metric.player_metrics()
    cols = ["distancia_m", "vel_max_kmh", "vel_media_kmh", "sprints", "dist_alta_int_m", "frames"]

    print(f"{'ID':>4} {'dist_m':>7} {'vmax':>6} {'vmed':>6} {'spr':>4} {'alta_m':>7}")
    for tid, m in sorted(metrics.items(), key=lambda x: -x[1]["distancia_m"]):
        print(f"{tid:>4} {m['distancia_m']:>7} {m['vel_max_kmh']:>6} "
              f"{m['vel_media_kmh']:>6} {m['sprints']:>4} {m['dist_alta_int_m']:>7}")

    out = path.with_name(path.stem.replace("_positions", "") + "_stats.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["track_id"] + cols)
        for tid, m in sorted(metrics.items(), key=lambda x: -x[1]["distancia_m"]):
            wr.writerow([tid] + [m[c] for c in cols])
    print(f"\n[OK] Stats guardadas en: {out}")


if __name__ == "__main__":
    main()

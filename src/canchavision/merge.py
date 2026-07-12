"""Fusión de rastros (track stitching) para arreglar la identidad.

El tracker parte a un mismo jugador en varios IDs cuando lo pierde un instante
(cruces, oclusiones). Esto une esos fragmentos: si un rastro A termina y otro B
empieza poco después, del mismo equipo, y en una posición alcanzable a velocidad
humana durante el hueco, se consideran el mismo jugador.

Trabaja sobre las posiciones ya guardadas (en metros), así que es instantáneo:
no hay que reprocesar el vídeo.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _summarize(rows) -> dict[int, dict]:
    tracks: dict[int, list] = defaultdict(list)
    for r in rows:
        tracks[int(r["track_id"])].append(
            (int(r["frame"]), int(r["team"]), float(r["x_m"]), float(r["y_m"]))
        )
    summ: dict[int, dict] = {}
    for tid, pts in tracks.items():
        pts.sort()
        teams = [p[1] for p in pts]
        summ[tid] = {
            "start_f": pts[0][0],
            "end_f": pts[-1][0],
            "start_xy": np.array([pts[0][2], pts[0][3]]),
            "end_xy": np.array([pts[-1][2], pts[-1][3]]),
            "team": max(set(teams), key=teams.count),
            "n": len(pts),
        }
    return summ


def merge_tracks(
    rows,
    fps: float = 25.0,
    max_gap_s: float = 1.5,
    max_speed_ms: float = 8.0,
    dist_tol_m: float = 2.0,
    same_team: bool = True,
) -> dict[int, int]:
    """Devuelve un mapeo {id_original: id_fusionado}."""
    summ = _summarize(rows)
    ids = sorted(summ)
    parent = {t: t for t in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    max_gap = max_gap_s * fps
    used_pred: set[int] = set()

    # Para cada rastro B (por orden de inicio), busca el mejor predecesor A libre.
    for b in sorted(ids, key=lambda t: summ[t]["start_f"]):
        sb = summ[b]
        best, best_cost = None, None
        for a in ids:
            if a == b or a in used_pred:
                continue
            sa = summ[a]
            gap = sb["start_f"] - sa["end_f"]
            if gap <= 0 or gap > max_gap:
                continue
            if same_team and sa["team"] != sb["team"]:
                continue
            dist = float(np.linalg.norm(sb["start_xy"] - sa["end_xy"]))
            if dist > max_speed_ms * (gap / fps) + dist_tol_m:
                continue
            cost = dist + gap / fps
            if best is None or cost < best_cost:
                best, best_cost = a, cost
        if best is not None:
            parent[find(b)] = find(best)
            used_pred.add(best)

    # Relabel de raíces a IDs pequeños y estables (por frame de inicio).
    roots = sorted({find(t) for t in ids}, key=lambda r: summ[r]["start_f"])
    root_to_new = {r: i + 1 for i, r in enumerate(roots)}
    return {t: root_to_new[find(t)] for t in ids}


def apply_mapping(rows, mapping: dict[int, int]) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        r["track_id"] = mapping.get(int(r["track_id"]), int(r["track_id"]))
        out.append(r)
    return out

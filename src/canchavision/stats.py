"""Cálculo de estadísticas por jugador y equipo (Fase 3).

Acumula las posiciones de cada jugador (idealmente en coordenadas métricas del
campo, provistas por pitch.PitchMapper) y deriva métricas. En Fase 1, sin
homografía, opera en píxeles como marcador de posición.

TODO Fase 3: velocidad, sprints, posesión por zonas, heatmaps, formaciones.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class StatsCollector:
    def __init__(self, fps: float = 25.0):
        self.fps = fps
        self.tracks: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def update(self, tracker_ids, positions) -> None:
        if tracker_ids is None:
            return
        for tid, pos in zip(tracker_ids, positions):
            if tid is not None:
                self.tracks[int(tid)].append((float(pos[0]), float(pos[1])))

    def distance_covered(self) -> dict[int, float]:
        """Distancia total por jugador (unidades = las de las posiciones)."""
        out: dict[int, float] = {}
        for tid, pts in self.tracks.items():
            p = np.array(pts)
            out[tid] = (
                float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
                if len(p) >= 2
                else 0.0
            )
        return out


class MetricStats:
    """Métricas físicas por jugador a partir de posiciones en la cancha, en
    METROS. La velocidad se calcula sobre una ventana temporal (no frame a
    frame) para ser robusta al ruido de detección/homografía, con un tope de
    velocidad realista.
    """

    def __init__(self, fps: float = 25.0):
        self.fps = fps
        self.tracks: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def update(self, tracker_ids, xy_m) -> None:
        """xy_m: posiciones en METROS (Nx2) alineadas con tracker_ids."""
        if tracker_ids is None:
            return
        for tid, p in zip(tracker_ids, xy_m):
            if tid is not None:
                self.tracks[int(tid)].append((float(p[0]), float(p[1])))

    @staticmethod
    def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
        if w <= 1 or len(arr) < w:
            return arr
        k = np.ones(w) / w
        xs = np.convolve(arr[:, 0], k, mode="valid")
        ys = np.convolve(arr[:, 1], k, mode="valid")
        return np.column_stack([xs, ys])

    @staticmethod
    def _count_sprints(mask, min_len: int) -> int:
        count, run = 0, 0
        for m in mask:
            if m:
                run += 1
            else:
                if run >= min_len:
                    count += 1
                run = 0
        if run >= min_len:
            count += 1
        return count

    def player_metrics(
        self,
        smooth: int = 7,              # ventana de suavizado de posición
        vel_window: int = 3,          # velocidad = desplazamiento en ±3 frames
        max_speed_ms: float = 10.0,   # ~36 km/h; por encima = ruido
        sprint_ms: float = 5.5,       # ~20 km/h = alta intensidad
    ) -> dict[int, dict]:
        min_sprint_frames = max(1, int(self.fps * 0.4))
        out: dict[int, dict] = {}
        for tid, pts in self.tracks.items():
            if len(pts) < smooth + 2 * vel_window + 1:
                continue
            sm = self._smooth(np.array(pts, dtype=float), smooth)
            n = len(sm)

            # Distancia: paso a paso, descartando teletransportes (ID reasignado)
            steps = np.linalg.norm(np.diff(sm, axis=0), axis=1)  # m/frame
            steps[steps * self.fps > max_speed_ms] = 0.0
            dist = float(steps.sum())

            # Velocidad: desplazamiento sobre ventana ±w => más estable
            w = vel_window
            speeds = np.zeros(n)
            for i in range(w, n - w):
                d = np.linalg.norm(sm[i + w] - sm[i - w])
                speeds[i] = d / (2 * w / self.fps)
            speeds = np.clip(speeds, 0.0, max_speed_ms)

            hi = speeds >= sprint_ms
            hi_steps = hi[: len(steps)]
            avg_ms = dist / (n / self.fps) if n else 0.0
            out[tid] = {
                "distancia_m": round(dist, 1),
                "vel_max_kmh": round(float(speeds.max()) * 3.6, 1),
                "vel_media_kmh": round(avg_ms * 3.6, 1),
                "sprints": self._count_sprints(hi, min_sprint_frames),
                "dist_alta_int_m": round(float(steps[hi_steps].sum()), 1),
                "frames": n,
            }
        return out

    def distances_m(self, **kw) -> dict[int, float]:
        return {tid: m["distancia_m"] for tid, m in self.player_metrics(**kw).items()}

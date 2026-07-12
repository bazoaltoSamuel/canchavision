"""Mapas de calor por equipo y formación media a partir del tracking.

Acumula la posición de los pies de cada jugador (centro inferior de la caja) a lo
largo del vídeo y produce dos imágenes:
  - heatmap: dónde se concentró cada equipo.
  - formación: la posición media de cada rastro (ID), que dibuja el "esquema".

Nota: en Fase 1 se trabaja en coordenadas de imagen (perspectiva de la cámara),
no en metros. La versión métrica (vista cenital real) llega con la homografía
(pitch.py, Fase 2).
"""
from __future__ import annotations

import cv2
import numpy as np

# BGR: equipo 0 rojo, equipo 1 azul, extra amarillo
TEAM_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 255)]


class HeatmapCollector:
    def __init__(self, width: int, height: int, num_teams: int = 2):
        self.w = width
        self.h = height
        self.num_teams = num_teams
        self.points: dict[int, list[tuple[float, float]]] = {}
        self.track_points: dict[int, list[tuple[float, float]]] = {}
        self.track_team: dict[int, dict[int, int]] = {}

    def update(self, boxes, team_labels, tracker_ids) -> None:
        for box, team, tid in zip(boxes, team_labels, tracker_ids):
            x1, y1, x2, y2 = box
            fx, fy = (float(x1) + float(x2)) / 2.0, float(y2)  # pies del jugador
            t = int(team)
            self.points.setdefault(t, []).append((fx, fy))
            if tid is not None:
                self.track_points.setdefault(int(tid), []).append((fx, fy))
                counts = self.track_team.setdefault(int(tid), {})
                counts[t] = counts.get(t, 0) + 1

    def _heat(self, pts) -> np.ndarray:
        heat = np.zeros((self.h, self.w), np.float32)
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < self.w and 0 <= yi < self.h:
                heat[yi, xi] += 1.0
        if heat.max() > 0:
            heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=self.h * 0.03)
            heat /= heat.max()
        return heat

    def render_heatmap(self, background: np.ndarray) -> np.ndarray:
        out = background.astype(np.float32) * 0.45
        for t in range(self.num_teams):
            heat = self._heat(self.points.get(t, []))
            color = np.array(TEAM_COLORS[t % len(TEAM_COLORS)], np.float32)
            out += heat[..., None] * color
        return np.clip(out, 0, 255).astype(np.uint8)

    def render_formation(self, background: np.ndarray, min_points: int = 10) -> np.ndarray:
        out = (background.astype(np.float32) * 0.4).astype(np.uint8)
        for tid, pts in self.track_points.items():
            if len(pts) < min_points:
                continue
            arr = np.array(pts)
            mx, my = int(arr[:, 0].mean()), int(arr[:, 1].mean())
            counts = self.track_team.get(tid, {})
            team = max(counts, key=counts.get) if counts else 0
            color = TEAM_COLORS[team % len(TEAM_COLORS)]
            cv2.circle(out, (mx, my), 10, color, -1)
            cv2.circle(out, (mx, my), 10, (255, 255, 255), 2)
            cv2.putText(
                out, f"#{tid}", (mx - 10, my - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
        return out

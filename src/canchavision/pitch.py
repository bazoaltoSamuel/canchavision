"""Calibración del campo y homografía imagen -> coordenadas métricas (Fase 2).

Objetivo: dado un conjunto de correspondencias entre puntos de la imagen y puntos
conocidos del campo (esquinas, círculo central, áreas...), calcular la homografía
que permite proyectar posiciones de jugadores a metros reales. Eso habilita
distancia recorrida, velocidad y mapas de calor fiables.

TODO Fase 2:
  - detección automática de líneas/keypoints del campo
  - vista cenital ("radar view")
"""
from __future__ import annotations

import cv2
import numpy as np


class PitchMapper:
    def __init__(self, length_m: float = 105, width_m: float = 68):
        self.length_m = length_m
        self.width_m = width_m
        self.H: np.ndarray | None = None

    def calibrate(self, image_points, pitch_points) -> np.ndarray:
        """image_points y pitch_points: listas de (x, y) correspondientes."""
        self.H, _ = cv2.findHomography(
            np.array(image_points, dtype=np.float32),
            np.array(pitch_points, dtype=np.float32),
        )
        return self.H

    def to_pitch(self, points) -> np.ndarray:
        if self.H is None:
            raise RuntimeError("PitchMapper no calibrado; llama a calibrate().")
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

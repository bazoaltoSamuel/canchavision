"""Clasificación de jugadores por equipo según el color de la camiseta.

Enfoque robusto (CPU-friendly):
  - Feature de color del torso enmascarando el CÉSPED (píxeles verdes), para que
    el fondo no contamine el color de la camiseta.
  - Tono (hue) tratado de forma CIRCULAR (cos/sin) — evita el salto 179->0.
  - El modelo K-Means se ajusta con muestras acumuladas de VARIOS frames (no de
    uno solo), lo que da centros de cluster mucho más estables.

Como el detector de fútbol ya separa portero/árbitro en sus propias clases, aquí
solo entran jugadores de campo -> los 2 clusters son los dos equipos, sin ruido.

Mejora futura (requiere GPU): embeddings visuales (SigLIP) en vez de color, para
kits parecidos o con patrones.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    def __init__(self, num_teams: int = 2):
        self.num_teams = num_teams
        self.kmeans: KMeans | None = None
        self.fitted = False
        self._pool: list[np.ndarray] = []
        self._pool_frames = 0

    @staticmethod
    def _jersey_feature(frame: np.ndarray, box) -> np.ndarray | None:
        x1, y1, x2, y2 = map(int, box)
        h, w = max(y2 - y1, 1), max(x2 - x1, 1)
        ty1, ty2 = max(y1 + int(0.15 * h), 0), max(y1 + int(0.50 * h), 0)
        tx1, tx2 = max(x1 + int(0.10 * w), 0), max(x2 - int(0.10 * w), 0)
        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
        hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        grass = (hue >= 35) & (hue <= 85) & (sat > 60) & (val > 40)
        keep = ~grass
        px = hsv[keep] if keep.sum() >= 10 else hsv
        # Tono ponderado por saturación: los píxeles blancos/oscuros (sat baja) no
        # inyectan color falso; para camisetas claras manda el brillo (val).
        hue_rad = px[:, 0] * (2 * np.pi / 180.0)  # OpenCV hue 0-180 -> rad
        wsat = px[:, 1] / 255.0
        wsum = float(wsat.sum()) + 1e-6
        return np.array([
            float((np.cos(hue_rad) * wsat).sum() / wsum),
            float((np.sin(hue_rad) * wsat).sum() / wsum),
            float(px[:, 1].mean() / 255.0),
            float(px[:, 2].mean() / 255.0),
        ])

    def _features(self, frame: np.ndarray, boxes) -> np.ndarray:
        feats = []
        for b in boxes:
            f = self._jersey_feature(frame, b)
            feats.append(f if f is not None else np.zeros(4, dtype=np.float32))
        return np.array(feats) if feats else np.zeros((0, 4), dtype=np.float32)

    def observe(self, frame: np.ndarray, boxes) -> None:
        """Acumula muestras de color para el ajuste (varios frames)."""
        if self.fitted:
            return
        for b in boxes:
            f = self._jersey_feature(frame, b)
            if f is not None:
                self._pool.append(f)
        self._pool_frames += 1

    def try_fit(self, min_samples: int = 30, min_frames: int = 10) -> None:
        if self.fitted:
            return
        if len(self._pool) >= min_samples and self._pool_frames >= min_frames:
            self.kmeans = KMeans(
                n_clusters=self.num_teams, n_init=10, random_state=42
            ).fit(np.array(self._pool))
            self.fitted = True

    def predict(self, frame: np.ndarray, boxes) -> np.ndarray:
        if not self.fitted or len(boxes) == 0:
            return np.zeros(len(boxes), dtype=int)
        return self.kmeans.predict(self._features(frame, boxes))

    def predict_with_conf(self, frame: np.ndarray, boxes):
        """Etiqueta de equipo + confianza [0..1] por caja.

        La confianza es el margen relativo entre las distancias a los dos
        centros de cluster: alto = claramente de un equipo; ~0 = ambiguo
        (oclusión, blur, color mezclado). Sirve para PONDERAR el voto temporal
        y que los frames dudosos casi no cuenten.
        """
        n = len(boxes)
        if not self.fitted or n == 0:
            return np.zeros(n, dtype=int), np.zeros(n, dtype=float)
        d = self.kmeans.transform(self._features(frame, boxes))  # (N, k)
        labels = d.argmin(axis=1)
        ds = np.sort(d, axis=1)
        margin = (ds[:, 1] - ds[:, 0]) / (ds[:, 1] + 1e-6)  # 0..1
        return labels.astype(int), margin.astype(float)


class TeamVoteTracker:
    """Estabiliza el equipo por track ID acumulando votos ponderados.

    En lugar de creer la clasificación de color de un solo frame (que parpadea),
    cada track acumula votos por equipo y devuelve SIEMPRE la mayoría actual.
    Es monótonamente estable y auto-corrige: unos pocos frames malos no vuelcan
    a un track que ya tiene historial claro.
    """

    def __init__(self, num_teams: int = 2):
        self.num_teams = num_teams
        self.votes: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(num_teams, dtype=float)
        )

    def update(self, track_ids, labels, weights) -> np.ndarray:
        out = []
        for tid, lab, w in zip(track_ids, labels, weights):
            if tid is None:
                out.append(int(lab))
                continue
            v = self.votes[int(tid)]
            v[int(lab)] += max(float(w), 1e-3)  # aun con conf~0 suma un mínimo
            out.append(int(np.argmax(v)))
        return np.array(out, dtype=int)

    def vote(self, track_id, label, weight: float = 1.0) -> None:
        """Registra un voto suelto (p. ej. equipo heredado del portero)."""
        if track_id is None:
            return
        self.votes[int(track_id)][int(label)] += max(float(weight), 1e-3)

    def team_of(self, track_id) -> int | None:
        v = self.votes.get(int(track_id)) if track_id is not None else None
        return int(np.argmax(v)) if v is not None and v.sum() > 0 else None

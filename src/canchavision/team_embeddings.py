"""Clasificación de equipos por EMBEDDINGS visuales (SigLIP) — pensado para GPU.

El clasificador por color ([[teams.py]] -> TeamClassifier) falla cuando los dos
equipos visten kits PARECIDOS (blanco vs blanco): el color no los separa. Este
usa el modelo SigLIP para comparar la APARIENCIA COMPLETA del jugador (patrón,
escudo, tono, sombras) y separa los equipos aunque el color sea casi igual.

Es pesado (una red neuronal por recorte) -> pensado para GPU (Colab). Expone la
MISMA interfaz que TeamClassifier (observe / try_fit / fitted / predict_with_conf)
para ser un reemplazo directo en el pipeline, combinado con TeamVoteTracker.

Envuelve `sports.common.team.TeamClassifier` (SigLIP -> UMAP -> KMeans) y añade
una CONFIANZA por recorte a partir de las distancias de KMeans, para ponderar el
voto temporal igual que hace la versión de color.
"""
from __future__ import annotations

import numpy as np


def _crop(frame, box):
    """Recorte válido de la caja (o None si es degenerado/fuera de cuadro)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


class SiglipTeamClassifier:
    def __init__(self, num_teams: int = 2, device: str | None = None,
                 max_fit_crops: int = 300):
        import torch
        from sports.common.team import TeamClassifier as _Siglip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model = _Siglip(device=device)
        self.num_teams = num_teams
        self.fitted = False
        self._pool: list[np.ndarray] = []
        self._pool_frames = 0
        self.max_fit_crops = max_fit_crops

    def observe(self, frame, boxes) -> None:
        """Acumula recortes de jugador para el ajuste (varios frames)."""
        if self.fitted:
            return
        for b in boxes:
            c = _crop(frame, b)
            if c is not None:
                self._pool.append(c)
        self._pool_frames += 1

    def try_fit(self, min_samples: int = 60, min_frames: int = 12) -> None:
        if self.fitted:
            return
        if len(self._pool) >= min_samples and self._pool_frames >= min_frames:
            crops = self._pool
            if len(crops) > self.max_fit_crops:  # submuestrea para un fit rápido
                sel = np.linspace(0, len(crops) - 1, self.max_fit_crops).astype(int)
                crops = [crops[i] for i in sel]
            self._model.fit(crops)
            self.fitted = True
            self._pool = []  # libera memoria de recortes

    def _labels_conf(self, crops):
        """Etiqueta + confianza [0..1] replicando predict con distancias KMeans."""
        feats = self._model.extract_features(crops)
        proj = self._model.reducer.transform(feats)
        d = self._model.cluster_model.transform(proj)  # (N, k) dist a centros
        labels = d.argmin(axis=1)
        ds = np.sort(d, axis=1)
        margin = (ds[:, 1] - ds[:, 0]) / (ds[:, 1] + 1e-6)
        return labels.astype(int), margin.astype(float)

    def predict_with_conf(self, frame, boxes):
        n = len(boxes)
        if not self.fitted or n == 0:
            return np.zeros(n, dtype=int), np.zeros(n, dtype=float)
        crops, valid = [], []
        for i, b in enumerate(boxes):
            c = _crop(frame, b)
            if c is not None:
                crops.append(c)
                valid.append(i)
        labels = np.zeros(n, dtype=int)
        conf = np.zeros(n, dtype=float)
        if crops:
            lab, cf = self._labels_conf(crops)
            for k, i in enumerate(valid):
                labels[i], conf[i] = lab[k], cf[k]
        return labels, conf

    def predict(self, frame, boxes) -> np.ndarray:
        return self.predict_with_conf(frame, boxes)[0]

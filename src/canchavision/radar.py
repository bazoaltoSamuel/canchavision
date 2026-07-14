"""Vista cenital (radar) y homografía del campo — Fase 2.

Usa el modelo de puntos de referencia del campo de Roboflow para calcular la
homografía imagen -> cancha real (120x70 m, en cm) y proyectar a los jugadores
sobre un pizarrón visto desde arriba.

Para cámara fija, la homografía se calcula UNA vez (`calibrate`) y se reutiliza,
evitando correr la detección del campo en cada frame (que es lo lento).
"""
from __future__ import annotations

import os

import numpy as np
import supervision as sv
from inference import get_model
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

FIELD_MODEL_ID = "football-field-detection-f07vi/14"

TEAM_COLORS = [sv.Color(255, 0, 0), sv.Color(0, 0, 255), sv.Color(255, 255, 0)]


class PitchRadar:
    def __init__(
        self,
        api_key: str | None = None,
        min_keypoints: int = 6,
        kp_conf: float = 0.5,
    ):
        self.config = SoccerPitchConfiguration()
        self.model = get_model(
            FIELD_MODEL_ID, api_key=api_key or os.environ.get("ROBOFLOW_API_KEY")
        )
        self.min_keypoints = min_keypoints
        self.kp_conf = kp_conf
        self.transformer: ViewTransformer | None = None
        self._base_pitch = draw_pitch(self.config)

    @property
    def calibrated(self) -> bool:
        return self.transformer is not None

    def calibrate(self, frame) -> bool:
        # Mapea cada keypoint a su vértice del campo por class_id (0..31).
        # Robusto: funciona aunque el modelo devuelva menos de 32 puntos
        # (distintas versiones de inference devuelven distinto número).
        result = self.model.infer(frame, confidence=0.3)[0]
        vertices = self.config.vertices
        src, dst = [], []
        for inst in getattr(result, "predictions", []) or []:
            for k in getattr(inst, "keypoints", []) or []:
                cid = int(k.class_id)
                if 0 <= cid < len(vertices) and float(k.confidence) > self.kp_conf:
                    src.append([float(k.x), float(k.y)])
                    dst.append(vertices[cid])
        if len(src) < self.min_keypoints:
            return False
        self.transformer = ViewTransformer(
            source=np.array(src, dtype=np.float32),
            target=np.array(dst, dtype=np.float32),
        )
        return True

    def to_pitch(self, foot_xy) -> np.ndarray:
        """Proyecta puntos de imagen (Nx2) a coordenadas de cancha en cm."""
        if not self.calibrated:
            raise RuntimeError("PitchRadar no calibrado.")
        if len(foot_xy) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return self.transformer.transform_points(np.asarray(foot_xy, np.float32))

    def pitch_size(self) -> tuple[int, int]:
        return self._base_pitch.shape[1], self._base_pitch.shape[0]

    def render(self, pitch_xy: np.ndarray, team_labels: np.ndarray) -> np.ndarray:
        pitch = self._base_pitch.copy()
        for t in range(2):
            pts = pitch_xy[team_labels == t] if len(pitch_xy) else pitch_xy
            if len(pts):
                pitch = draw_points_on_pitch(
                    self.config, pts,
                    face_color=TEAM_COLORS[t], edge_color=sv.Color.WHITE,
                    radius=16, pitch=pitch,
                )
        return pitch

"""Dibujo de anotaciones sobre el frame: cajas, IDs, equipos y balón."""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

# Colores en BGR
TEAM_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 255)]  # rojo, azul, amarillo
BALL_COLOR = (0, 255, 0)


class Annotator:
    def draw(
        self,
        frame: np.ndarray,
        players: sv.Detections,
        team_labels: np.ndarray,
        balls: sv.Detections,
    ) -> np.ndarray:
        tracker_ids = (
            players.tracker_id
            if players.tracker_id is not None
            else [None] * len(players)
        )
        for box, tid, team in zip(players.xyxy, tracker_ids, team_labels):
            x1, y1, x2, y2 = map(int, box)
            color = TEAM_COLORS[int(team) % len(TEAM_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{int(tid)}" if tid is not None else "?"
            cv2.putText(
                frame, label, (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        for box in balls.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), max((x2 - x1) // 2, 5), BALL_COLOR, 2)

        return frame

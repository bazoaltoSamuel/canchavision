"""Demo Fase 2: vista cenital (radar) de un frame.

Detecta los puntos de referencia del campo, calcula la homografía y proyecta a
los jugadores sobre un campo visto desde arriba, coloreados por equipo.

Uso:
    ROBOFLOW_API_KEY=... python scripts/radar_demo.py <video> [frame_idx]
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import supervision as sv
from inference import get_model

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

from canchavision.teams import TeamClassifier

PLAYER_MODEL_ID = "football-players-detection-3zvbc/11"
FIELD_MODEL_ID = "football-field-detection-f07vi/14"
PLAYER_CLASS = 2

RED = sv.Color(255, 0, 0)
BLUE = sv.Color(0, 0, 255)


def main() -> None:
    video = sys.argv[1] if len(sys.argv) > 1 else "data/raw/2e57b9_0.mp4"
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("No se pudo leer el frame")

    config = SoccerPitchConfiguration()
    player_model = get_model(PLAYER_MODEL_ID, api_key=api_key)
    field_model = get_model(FIELD_MODEL_ID, api_key=api_key)

    # 1) Puntos de referencia del campo -> homografía imagen->cancha
    kp = sv.KeyPoints.from_inference(field_model.infer(frame, confidence=0.3)[0])
    mask = kp.confidence[0] > 0.5
    src = kp.xy[0][mask].astype(np.float32)
    dst = np.array(config.vertices, dtype=np.float32)[mask]
    print(f"Puntos de campo detectados: {int(mask.sum())} / 32")
    transformer = ViewTransformer(source=src, target=dst)

    # 2) Jugadores -> punto de los pies -> proyección a la cancha
    det = sv.Detections.from_inference(player_model.infer(frame, confidence=0.3)[0])
    players = det[det.class_id == PLAYER_CLASS]
    boxes = players.xyxy
    foot = np.array([[(x1 + x2) / 2, y2] for x1, y1, x2, y2 in boxes], np.float32)
    pitch_xy = transformer.transform_points(foot)

    # 3) Separar equipos por color
    teams = TeamClassifier(num_teams=2)
    teams.fit(frame, boxes)
    labels = teams.predict(frame, boxes)

    # 4) Dibujar radar cenital
    pitch = draw_pitch(config)
    pitch = draw_points_on_pitch(
        config, pitch_xy[labels == 0], face_color=RED,
        edge_color=sv.Color.WHITE, radius=16, pitch=pitch,
    )
    pitch = draw_points_on_pitch(
        config, pitch_xy[labels == 1], face_color=BLUE,
        edge_color=sv.Color.WHITE, radius=16, pitch=pitch,
    )

    out = "outputs/radar_demo.png"
    cv2.imwrite(out, pitch)
    print(f"[OK] Radar cenital guardado en: {out}  ({len(boxes)} jugadores)")


if __name__ == "__main__":
    main()

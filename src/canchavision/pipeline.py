"""Orquestación del pipeline completo sobre un vídeo (Fase 1)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import csv

from .annotate import Annotator
from .detect import Detector, RoboflowDetector
from .heatmap import HeatmapCollector
from .stats import MetricStats
from .teams import TeamClassifier, TeamVoteTracker
from .track import Tracker


def run_pipeline(
    video_path: str | Path,
    cfg: dict,
    output_path: str | Path | None = None,
    max_frames: int | None = None,
    start_sec: float = 0.0,
) -> Path:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"No se encontró el vídeo: {video_path}")

    if output_path is None:
        output_path = Path("outputs") / f"{video_path.stem}_annotated.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.get("detector", "yolo") == "roboflow":
        detector = RoboflowDetector(**cfg["model"])
    else:
        detector = Detector(**cfg["model"])
    teams = TeamClassifier(num_teams=cfg["teams"]["num_teams"])
    team_votes = TeamVoteTracker(num_teams=cfg["teams"]["num_teams"])
    annotator = Annotator()

    player_id = cfg["classes"].get("player", cfg["classes"].get("person"))
    gk_id = cfg["classes"].get("goalkeeper")
    track_class_ids = [player_id] + ([gk_id] if gk_id is not None else [])
    ball_id = cfg["classes"]["ball"]
    fit_min = cfg["teams"].get("fit_min_players", 4)
    min_box_height = cfg.get("detect_filter", {}).get("min_box_height", 0)
    recal_every = cfg.get("radar_recalibrate_every", 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV no pudo abrir el vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = Tracker(frame_rate=int(round(fps)), **cfg.get("tracker", {}))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
        total = max(total - int(start_sec * fps), 0)

    if max_frames:
        total = min(total, max_frames) if total > 0 else max_frames

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    heat = HeatmapCollector(w, h, cfg["teams"]["num_teams"])
    background = None

    # --- Radar cenital + distancias (Fase 2, opcional) ---
    radar_enabled = cfg.get("radar", False)
    radar = None
    radar_writer = None
    metric = MetricStats(fps=fps)
    positions_rows: list[tuple] = []  # frame, track_id, team, x_m, y_m
    ball_rows: list[tuple] = []       # frame, x_m, y_m, conf
    if radar_enabled:
        from .radar import PitchRadar

        radar = PitchRadar()
        rw, rh = radar.pitch_size()
        radar_path = output_path.with_name(f"{video_path.stem}_radar.mp4")
        radar_writer = cv2.VideoWriter(
            str(radar_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (rw, rh)
        )

    idx = 0
    pbar = tqdm(total=total or None, desc="Procesando")
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break

        det = detector.detect(frame)
        players = det[np.isin(det.class_id, track_class_ids)]
        if min_box_height > 0 and len(players) > 0:
            heights = players.xyxy[:, 3] - players.xyxy[:, 1]
            players = players[heights >= min_box_height]
        players = tracker.update(players)
        balls = det[det.class_id == ball_id]

        boxes = players.xyxy
        outfield = players.class_id == player_id  # jugadores de campo (no portero)
        tracker_ids = (
            players.tracker_id
            if players.tracker_id is not None
            else [None] * len(boxes)
        )
        if not teams.fitted:
            teams.observe(frame, boxes[outfield] if len(boxes) else boxes)
            teams.try_fit(min_frames=min(10, fit_min * 2))

        # Equipo ESTABLE por track: voto temporal ponderado por confianza de color.
        # El portero no vota por color (su kit difiere de ambos equipos): peso 0
        # aquí y se resuelve por herencia + voto justo debajo.
        raw_labels, conf = teams.predict_with_conf(frame, boxes)
        w = conf.copy()
        if gk_id is not None and len(w):
            w[~outfield] = 0.0
        team_labels = team_votes.update(tracker_ids, raw_labels, w)

        # Portero: hereda el equipo (ya estable) del jugador de campo más cercano
        # y lo vota, de modo que converge y deja de parpadear con el tiempo.
        if gk_id is not None and teams.fitted and len(boxes) and outfield.any() and (~outfield).any():
            ofc = np.column_stack([
                (boxes[outfield][:, 0] + boxes[outfield][:, 2]) / 2,
                (boxes[outfield][:, 1] + boxes[outfield][:, 3]) / 2,
            ])
            oft = team_labels[outfield]
            for i in np.where(~outfield)[0]:
                cx, cy = (boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2
                j = int(np.argmin((ofc[:, 0] - cx) ** 2 + (ofc[:, 1] - cy) ** 2))
                inferred = int(oft[j])
                team_votes.vote(tracker_ids[i], inferred, weight=0.5)
                resolved = team_votes.team_of(tracker_ids[i])
                team_labels[i] = resolved if resolved is not None else inferred

        if background is None:
            background = frame.copy()
        heat.update(boxes, team_labels, tracker_ids)

        if radar_enabled and len(boxes) > 0:
            foot = np.column_stack([
                (boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]
            ]).astype(np.float32)
            if (not radar.calibrated) or (recal_every > 0 and idx % recal_every == 0):
                radar.calibrate(frame)  # recalibra (mejor ubicación; aguanta paneos)
            if radar.calibrated:
                pitch_xy = radar.to_pitch(foot)      # cm
                xy_m = pitch_xy / 100.0              # metros
                metric.update(tracker_ids, xy_m)
                for tid, team, (x, y) in zip(tracker_ids, team_labels, xy_m):
                    if tid is not None:
                        positions_rows.append(
                            (idx, int(tid), int(team), round(float(x), 2), round(float(y), 2))
                        )
                ball_pitch = None
                if len(balls) > 0:
                    bi = int(np.argmax(balls.confidence))
                    bx1, by1, bx2, by2 = balls.xyxy[bi]
                    bc = np.array([[(bx1 + bx2) / 2.0, (by1 + by2) / 2.0]], dtype=np.float32)
                    ball_pitch = radar.to_pitch(bc)   # cm, shape (1, 2)
                    bm = ball_pitch[0] / 100.0
                    ball_rows.append(
                        (idx, round(float(bm[0]), 2), round(float(bm[1]), 2),
                         round(float(balls.confidence[bi]), 2))
                    )
                radar_writer.write(radar.render(pitch_xy, team_labels, ball_pitch))

        frame = annotator.draw(frame, players, team_labels, balls)
        writer.write(frame)

        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"\n[OK] Vídeo anotado guardado en: {output_path}")

    if background is not None:
        heat_path = output_path.with_name(f"{video_path.stem}_heatmap.png")
        form_path = output_path.with_name(f"{video_path.stem}_formation.png")
        cv2.imwrite(str(heat_path), heat.render_heatmap(background))
        cv2.imwrite(str(form_path), heat.render_formation(background))
        print(f"[OK] Mapa de calor:  {heat_path}")
        print(f"[OK] Formación:      {form_path}")

    if radar_writer is not None:
        radar_writer.release()
        radar_path = output_path.with_name(f"{video_path.stem}_radar.mp4")
        print(f"[OK] Radar cenital:  {radar_path}")

        if positions_rows:
            pos_path = output_path.with_name(f"{video_path.stem}_positions.csv")
            with open(pos_path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["frame", "track_id", "team", "x_m", "y_m"])
                wr.writerows(positions_rows)
            print(f"[OK] Posiciones:     {pos_path}  ({len(positions_rows)} filas)")

        if ball_rows:
            ball_path = output_path.with_name(f"{video_path.stem}_ball.csv")
            with open(ball_path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["frame", "x_m", "y_m", "conf"])
                wr.writerows(ball_rows)
            print(f"[OK] Balón:          {ball_path}  ({len(ball_rows)} frames)")

        metrics = metric.player_metrics()
        if metrics:
            cols = ["distancia_m", "vel_max_kmh", "vel_media_kmh",
                    "sprints", "dist_alta_int_m", "frames"]
            stats_path = output_path.with_name(f"{video_path.stem}_stats.csv")
            with open(stats_path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["track_id"] + cols)
                for tid, m in sorted(metrics.items(),
                                     key=lambda x: -x[1]["distancia_m"]):
                    wr.writerow([tid] + [m[c] for c in cols])
            print(f"[OK] Stats jugadores: {stats_path}")
            vmax = max(m["vel_max_kmh"] for m in metrics.values())
            tot_sprints = sum(m["sprints"] for m in metrics.values())
            print(f"     Vel. máx registrada: {vmax} km/h | sprints totales: {tot_sprints}")

    return output_path

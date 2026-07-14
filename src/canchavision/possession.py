"""Posesión y pases a partir de posiciones de jugadores y del balón (en metros).

Lógica (v1):
  - Por frame, el poseedor es el jugador más cercano al balón dentro de un radio.
  - Se suaviza (voto mayoritario en una ventana) para puentear frames sin balón y
    quitar parpadeos.
  - Se comprime en "posesiones" (spells) consecutivas por jugador.
  - Cada cambio de poseedor = un pase del jugador anterior:
        mismo equipo  -> pase COMPLETADO
        equipo rival / balón perdido -> pase FALLIDO
  - Posesión % = fracción de frames con balón controlado por cada equipo.

Trabaja sobre datos ya persistidos, así que es instantáneo y re-ajustable.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict


def load_players(path):
    by_frame = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_frame[int(r["frame"])].append(
                (int(r["track_id"]), int(r["team"]), float(r["x_m"]), float(r["y_m"]))
            )
    return by_frame


def load_ball(path):
    by_frame = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_frame[int(r["frame"])] = (float(r["x_m"]), float(r["y_m"]))
    return by_frame


def interpolate_ball(ball_by_frame, frames, max_gap=10):
    """Rellena los frames sin balón interpolando entre posiciones conocidas,
    siempre que el hueco no sea demasiado grande (balón realmente perdido)."""
    known = sorted(f for f in frames if f in ball_by_frame)
    out = dict(ball_by_frame)
    for a, b in zip(known, known[1:]):
        gap = b - a
        if 1 < gap <= max_gap:
            (xa, ya), (xb, yb) = ball_by_frame[a], ball_by_frame[b]
            for k in range(1, gap):
                t = k / gap
                out[a + k] = (xa + (xb - xa) * t, ya + (yb - ya) * t)
    return out


def _nearest_possessor(players, ball, max_dist_m):
    bx, by = ball
    best, best_d = None, None
    for tid, team, x, y in players:
        d = ((x - bx) ** 2 + (y - by) ** 2) ** 0.5
        if best_d is None or d < best_d:
            best_d, best = d, (tid, team)
    if best is not None and best_d <= max_dist_m:
        return best
    return None


def compute(players_by_frame, ball_by_frame, max_dist_m=3.0,
            smooth=5, min_spell=3, interp_gap=10):
    frames = sorted(players_by_frame)

    # 0) Interpolar el balón para puentear frames perdidos
    ball_by_frame = interpolate_ball(ball_by_frame, frames, max_gap=interp_gap)

    # 1) Poseedor crudo por frame
    raw = {}
    for f in frames:
        ball = ball_by_frame.get(f)
        raw[f] = _nearest_possessor(players_by_frame[f], ball, max_dist_m) if ball else None

    # Equipo de cada rastro = mayoría GLOBAL en todo el clip. Decidirlo una sola
    # vez (y no frame a frame) evita que un error puntual de etiquetado de equipo
    # convierta un pase correcto en "fallido".
    team_votes: dict[int, Counter] = defaultdict(Counter)
    for f in frames:
        for tid, team, x, y in players_by_frame[f]:
            team_votes[tid][team] += 1
    team_of = {tid: v.most_common(1)[0][0] for tid, v in team_votes.items()}

    # 2) Suavizado: voto mayoritario de track_id en ventana centrada
    smoothed = {}
    for i, f in enumerate(frames):
        window = [raw[frames[j]] for j in range(max(0, i - smooth), min(len(frames), i + smooth + 1))]
        votes = Counter(p[0] for p in window if p is not None)
        smoothed[f] = votes.most_common(1)[0][0] if votes else None

    # 3) Posesión % por equipo
    team_frames = Counter()
    for f in frames:
        tid = smoothed[f]
        if tid is not None:
            team_frames[team_of[tid]] += 1
    total_owned = sum(team_frames.values()) or 1
    possession_pct = {t: round(100 * c / total_owned, 1) for t, c in team_frames.items()}

    # 4) Spells (posesiones consecutivas por jugador), filtrando las muy cortas
    spells = []
    for f in frames:
        tid = smoothed[f]
        if tid is None:
            continue
        if spells and spells[-1][0] == tid:
            spells[-1][2] = f  # extiende fin
        else:
            spells.append([tid, f, f])  # [tid, start, end]
    spells = [s for s in spells if (s[2] - s[1] + 1) >= min_spell]

    # 5) Pases: cada cambio de poseedor
    passes = defaultdict(lambda: {"completados": 0, "fallidos": 0})
    for a, b in zip(spells, spells[1:]):
        passer, receiver = a[0], b[0]
        if passer == receiver:
            continue
        if team_of.get(passer) == team_of.get(receiver):
            passes[passer]["completados"] += 1
        else:
            passes[passer]["fallidos"] += 1

    return {
        "possession_pct": possession_pct,
        "passes": {tid: dict(v) for tid, v in passes.items()},
        "team_of": team_of,
        "n_spells": len(spells),
        "frames_con_poseedor": total_owned,
        "frames_totales": len(frames),
    }

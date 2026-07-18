"""Diagnóstico de la asignación de equipos a partir de un _positions.csv.

Responde SIN adivinar a: ¿por qué un jugador parece cambiar de equipo?
  - ¿Cuántos tracks hay y cómo están balanceados los equipos?
  - ¿Hay tracks cuyo equipo PARPADEA dentro del propio track? (fallo de voto)
  - ¿Hay FRAGMENTACIÓN: dos tracks contiguos en tiempo/espacio con equipos
    distintos (mismo jugador partido en dos IDs, cada uno un equipo)?

Uso:
    python scripts/diagnose_teams.py outputs/<video>_positions.csv
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict

import numpy as np


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))

    per_track = defaultdict(list)  # tid -> [(frame, team, x, y)]
    frame_team = Counter()
    for r in rows:
        tid = int(r["track_id"])
        team = int(r["team"])
        per_track[tid].append(
            (int(r["frame"]), team, float(r["x_m"]), float(r["y_m"]))
        )
        frame_team[team] += 1

    n_tracks = len(per_track)
    print(f"Tracks: {n_tracks}  |  detecciones por equipo: {dict(frame_team)}")

    # 1) Equipo mayoritario y ESTABILIDAD por track
    team_of, unstable = {}, []
    for tid, pts in per_track.items():
        teams = [p[1] for p in pts]
        c = Counter(teams)
        team_of[tid] = c.most_common(1)[0][0]
        minority = sum(v for k, v in c.items() if k != team_of[tid])
        frac = minority / len(teams)
        if frac > 0.10:  # >10% de frames en el otro equipo = track inestable
            unstable.append((tid, round(frac, 2), dict(c)))

    print(f"\nTracks por equipo (voto global): {Counter(team_of.values())}")
    print(f"Tracks INESTABLES (parpadean >10% dentro del track): {len(unstable)}")
    for tid, frac, c in sorted(unstable, key=lambda x: -x[1])[:10]:
        print(f"   ID {tid:>3}: {int(frac*100)}% en el otro equipo  {c}")

    # 2) FRAGMENTACIÓN: pares de tracks que parecen el mismo jugador
    #    (uno termina, otro empieza poco después y cerca) pero con equipo distinto.
    summ = {}
    for tid, pts in per_track.items():
        pts.sort()
        summ[tid] = {
            "s": pts[0][0], "e": pts[-1][0],
            "sxy": np.array(pts[0][2:]), "exy": np.array(pts[-1][2:]),
            "team": team_of[tid],
        }
    frag_same, frag_diff = [], []
    for a in summ:
        for b in summ:
            if a == b:
                continue
            gap = summ[b]["s"] - summ[a]["e"]
            if 0 < gap <= 30:  # ~1 s
                dist = float(np.linalg.norm(summ[b]["sxy"] - summ[a]["exy"]))
                if dist <= 8.0:  # alcanzable a velocidad humana
                    if summ[a]["team"] == summ[b]["team"]:
                        frag_same.append((a, b))
                    else:
                        frag_diff.append((a, b, summ[a]["team"], summ[b]["team"]))

    print(f"\nPosible fragmentación (un track sigue a otro cercano en tiempo/espacio):")
    print(f"   mismo equipo : {len(frag_same)}")
    print(f"   EQUIPO DISTINTO (esto causa el 'cambia de equipo'): {len(frag_diff)}")
    for a, b, ta, tb in frag_diff[:10]:
        print(f"   ID {a}(eq{ta}) -> ID {b}(eq{tb})")

    # 3) Veredicto
    print("\n=== VEREDICTO ===")
    if len(frag_diff) >= 3:
        print("Causa dominante: FRAGMENTACIÓN de ID (mismo jugador partido en 2 IDs")
        print("con equipos distintos). Arreglo: fusionar tracks ANTES de asignar")
        print("equipo, o asignar equipo por apariencia agregada del jugador.")
    elif len(unstable) >= 3:
        print("Causa dominante: CLASIFICACIÓN inestable dentro del track. Arreglo:")
        print("reforzar el voto / usar SigLIP / feature por brillo si kits B/N.")
    else:
        print("Equipos razonablemente estables. El error de pases puede venir de")
        print("pocos casos puntuales; revisar los spells concretos del jugador.")


if __name__ == "__main__":
    main()

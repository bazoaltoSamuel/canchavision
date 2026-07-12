"""Calcula posesión y pases desde los CSV de posiciones y balón (instantáneo).

Uso:
    python scripts/compute_possession.py outputs/<v>_positions.csv outputs/<v>_ball.csv
"""
from __future__ import annotations

import sys

from canchavision import possession as P


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    players = P.load_players(sys.argv[1])
    ball = P.load_ball(sys.argv[2])

    res = P.compute(players, ball)

    print(f"Frames: {res['frames_totales']} | con poseedor claro: "
          f"{res['frames_con_poseedor']} | posesiones: {res['n_spells']}\n")

    print("== POSESIÓN POR EQUIPO ==")
    names = {0: "Equipo A (rojo)", 1: "Equipo B (azul)"}
    for team, pct in sorted(res["possession_pct"].items()):
        print(f"  {names.get(team, team)}: {pct}%")

    print("\n== PASES POR JUGADOR ==")
    print(f"{'ID':>4} {'equipo':>7} {'compl':>6} {'fall':>5} {'%acierto':>9}")
    passes = res["passes"]
    for tid in sorted(passes, key=lambda t: -(passes[t]['completados'] + passes[t]['fallidos'])):
        c = passes[tid]["completados"]
        fa = passes[tid]["fallidos"]
        tot = c + fa
        acc = f"{100*c/tot:.0f}%" if tot else "-"
        team = res["team_of"].get(tid, "?")
        print(f"{tid:>4} {names.get(team, team).split()[0]:>7} {c:>6} {fa:>5} {acc:>9}")

    tot_c = sum(p["completados"] for p in passes.values())
    tot_f = sum(p["fallidos"] for p in passes.values())
    tot = tot_c + tot_f or 1
    print(f"\nTotal pases: {tot_c + tot_f} | completados {tot_c} "
          f"({100*tot_c/tot:.0f}%) | fallidos {tot_f}")


if __name__ == "__main__":
    main()

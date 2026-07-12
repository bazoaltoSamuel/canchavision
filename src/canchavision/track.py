"""Seguimiento multi-objeto: asigna un ID estable a cada jugador."""
from __future__ import annotations

import supervision as sv


class Tracker:
    def __init__(
        self,
        frame_rate: int = 25,
        lost_track_buffer: int = 60,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.8,
    ):
        # lost_track_buffer alto = recuerda un rastro perdido más tiempo antes de
        # asignar un ID nuevo -> menos saltos de identidad en cruces/oclusiones.
        self.tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            lost_track_buffer=lost_track_buffer,
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        self.tracker.reset()

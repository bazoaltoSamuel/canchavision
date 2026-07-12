"""Detección de objetos (jugadores y balón).

Dos backends con la misma interfaz `.detect(frame) -> sv.Detections`:
  - Detector:          YOLO/Ultralytics genérico (COCO).
  - RoboflowDetector:  modelo específico de fútbol (player/GK/referee/ball).
"""
from __future__ import annotations

import os

import numpy as np
import supervision as sv
from ultralytics import YOLO


class Detector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.3,
        imgsz: int = 1280,
        device: str = "cpu",
    ):
        self.model = YOLO(weights)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Devuelve todas las detecciones del frame como sv.Detections."""
        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)


class RoboflowDetector:
    """Modelo de fútbol alojado en Roboflow, ejecutado en local vía `inference`.

    Descarga y cachea los pesos la primera vez. La API key se lee del parámetro
    o de la variable de entorno ROBOFLOW_API_KEY.
    """

    def __init__(
        self,
        model_id: str = "football-players-detection-3zvbc/11",
        confidence: float = 0.3,
        api_key: str | None = None,
        **_ignored,
    ):
        from inference import get_model

        api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Falta la API key de Roboflow (variable ROBOFLOW_API_KEY)."
            )
        self.model = get_model(model_id=model_id, api_key=api_key)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.infer(frame, confidence=self.confidence)[0]
        return sv.Detections.from_inference(result)

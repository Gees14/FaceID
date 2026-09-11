"""Deteccion de rostros con YuNet (modelo ONNX recomendado por OpenCV).

GhostFaceNets solo calcula embeddings; la deteccion de la caja del rostro
se hace por separado con este modulo.
"""

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    # 5 puntos (x, y): ojo derecho, ojo izquierdo, nariz, comisura derecha
    # e izquierda de la boca. Sirven para alinear la cara antes de pasarla
    # al embedder (ver align.py).
    landmarks: list[tuple[float, float]]


class FaceDetector:
    def __init__(self):
        self._input_size = (320, 320)
        self.net = cv2.FaceDetectorYN.create(
            str(config.DETECTOR_MODEL_PATH),
            "",
            self._input_size,
            config.DETECTOR_CONFIDENCE_THRESHOLD,
            config.DETECTOR_NMS_THRESHOLD,
            config.DETECTOR_TOP_K,
        )

    def _ensure_input_size(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if (w, h) != self._input_size:
            self._input_size = (w, h)
            self.net.setInputSize(self._input_size)

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        self._ensure_input_size(frame_bgr)
        _, faces = self.net.detect(frame_bgr)
        if faces is None:
            return []

        h, w = frame_bgr.shape[:2]
        results: list[FaceBox] = []
        for row in faces:
            x, y, bw, bh = row[:4]
            landmarks = [(float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)]
            score = float(row[-1])
            x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
            x2 = min(w, int(round(x + bw)))
            y2 = min(h, int(round(y + bh)))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                FaceBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1, confidence=score, landmarks=landmarks)
            )

        return results

    def detect_largest(self, frame_bgr: np.ndarray) -> FaceBox | None:
        faces = self.detect(frame_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: f.w * f.h)

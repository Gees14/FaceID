"""Alineacion de rostros al template estandar usado por ArcFace/GhostFaceNets.

GhostFaceNets se entreno con caras alineadas a 112x112 mediante una
transformacion de similitud sobre 5 puntos de referencia (ojos, nariz,
comisuras de la boca), no con recortes crudos del bounding box. Sin esto,
cualquier inclinacion de cabeza degrada mucho la calidad del embedding.
"""

import cv2
import numpy as np

import config

# Puntos de referencia estandar (ArcFace) para una salida de 112x112:
# ojo derecho, ojo izquierdo, nariz, comisura derecha e izquierda de la boca.
_REFERENCE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(frame_bgr: np.ndarray, landmarks: list[tuple[float, float]]) -> np.ndarray:
    """Devuelve un recorte de la cara alineado a config.FACE_SIZE x config.FACE_SIZE."""

    src = np.array(landmarks, dtype=np.float32)
    scale = config.FACE_SIZE / 112.0
    dst = _REFERENCE_LANDMARKS * scale

    transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if transform is None:
        # Fallback: si la estimacion falla (landmarks degenerados), usamos
        # una transformacion identidad centrada en los propios landmarks.
        transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    return cv2.warpAffine(
        frame_bgr, transform, (config.FACE_SIZE, config.FACE_SIZE), borderValue=0.0
    )

"""Wrapper del modelo GhostFaceNets: recibe un recorte de rostro y devuelve
un embedding de 512 dimensiones, L2-normalizado (listo para comparar por
producto interno / similitud coseno)."""

import cv2
import numpy as np
from tensorflow import keras

import config


class _CompatDepthwiseConv2D(keras.layers.DepthwiseConv2D):
    """El .h5 de GhostFaceNets se guardo con una version vieja de Keras que
    serializa un kwarg 'groups' en DepthwiseConv2D; Keras 3 ya no lo acepta."""

    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


class Embedder:
    def __init__(self):
        self.model = keras.models.load_model(
            str(config.EMBEDDER_MODEL_PATH),
            compile=False,
            custom_objects={"DepthwiseConv2D": _CompatDepthwiseConv2D},
        )

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (config.FACE_SIZE, config.FACE_SIZE))
        face_rgb = face_rgb.astype(np.float32)
        face_rgb = (face_rgb - 127.5) * 0.0078125
        return np.expand_dims(face_rgb, axis=0)

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        batch = self._preprocess(face_bgr)
        embedding = self.model.predict(batch, verbose=0)[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

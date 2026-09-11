"""Base vectorial local con FAISS: guarda un embedding de 512d por persona
registrada y permite buscar la mas parecida por similitud coseno (producto
interno, ya que los embeddings vienen L2-normalizados). Se persiste en disco
como un indice FAISS + un labels.json (id -> nombre)."""

import json
from typing import Optional

import faiss
import numpy as np

import config


class VectorDB:
    def __init__(self):
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(config.EMBEDDING_SIZE))
        self.labels: dict[int, str] = {}
        self._next_id = 0
        self._load()

    def _load(self) -> None:
        if config.FAISS_INDEX_PATH.exists():
            self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        if config.LABELS_PATH.exists():
            with open(config.LABELS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.labels = {int(k): v for k, v in raw.items()}
        if self.labels:
            self._next_id = max(self.labels.keys()) + 1

    def _save(self) -> None:
        faiss.write_index(self.index, str(config.FAISS_INDEX_PATH))
        with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)

    def add(self, name: str, embedding: np.ndarray) -> int:
        new_id = self._next_id
        self._next_id += 1

        vector = np.expand_dims(embedding.astype(np.float32), axis=0)
        ids = np.array([new_id], dtype=np.int64)
        self.index.add_with_ids(vector, ids)
        self.labels[new_id] = name
        self._save()
        return new_id

    def search(self, embedding: np.ndarray, k: int = 1) -> tuple[Optional[str], float]:
        if self.index.ntotal == 0:
            return None, 0.0

        vector = np.expand_dims(embedding.astype(np.float32), axis=0)
        scores, ids = self.index.search(vector, k)
        best_score = float(scores[0][0])
        best_id = int(ids[0][0])

        if best_id == -1 or best_score < config.MATCH_THRESHOLD:
            return None, best_score

        return self.labels.get(best_id, "Desconocido"), best_score

    def list_people(self) -> list[dict]:
        return [{"id": person_id, "name": name} for person_id, name in sorted(self.labels.items())]

    def delete(self, person_id: int) -> bool:
        if person_id not in self.labels:
            return False

        self.index.remove_ids(np.array([person_id], dtype=np.int64))
        del self.labels[person_id]
        self._save()
        return True

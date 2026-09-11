"""Backend FastAPI de la POC de FaceID.

- WS  /ws/stream   -> recibe frames de la camara, devuelve caras detectadas + nombre reconocido
- POST /enroll     -> registra una persona nueva a partir de una imagen
- GET  /people     -> lista personas registradas
- DELETE /people/{person_id} -> elimina una persona registrada
"""

import base64

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import download_models
from align import align_face
from embedder import Embedder
from face_detector import FaceDetector
from vector_db import VectorDB

app = FastAPI(title="FaceID POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: FaceDetector | None = None
embedder: Embedder | None = None
vector_db: VectorDB | None = None


@app.on_event("startup")
def load_models() -> None:
    global detector, embedder, vector_db

    if not (config.EMBEDDER_MODEL_PATH.exists() and config.DETECTOR_MODEL_PATH.exists()):
        print("Faltan modelos, descargando...")
        download_models.main()

    detector = FaceDetector()
    embedder = Embedder()
    vector_db = VectorDB()
    print("Modelos cargados. Backend listo.")


def decode_base64_image(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("No se pudo decodificar la imagen")
    return frame




@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") != "frame":
                continue

            try:
                frame = decode_base64_image(payload["image"])
            except Exception:
                await websocket.send_json({"type": "error", "message": "imagen invalida"})
                continue

            faces = detector.detect(frame)
            results = []
            for box in faces:
                aligned_face = align_face(frame, box.landmarks)
                embedding = embedder.get_embedding(aligned_face)
                name, score = vector_db.search(embedding)
                results.append(
                    {
                        "box": [box.x, box.y, box.w, box.h],
                        "name": name,
                        "score": round(score, 3),
                    }
                )

            await websocket.send_json({"type": "detections", "faces": results})
    except WebSocketDisconnect:
        pass


class EnrollRequest(BaseModel):
    name: str
    image: str


@app.post("/enroll")
def enroll(req: EnrollRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="El nombre no puede estar vacio")

    try:
        frame = decode_base64_image(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen invalida")

    box = detector.detect_largest(frame)
    if box is None:
        raise HTTPException(status_code=422, detail="No se detecto ningun rostro en la imagen")

    aligned_face = align_face(frame, box.landmarks)
    embedding = embedder.get_embedding(aligned_face)
    person_id = vector_db.add(name, embedding)

    return {"id": person_id, "name": name}


@app.get("/people")
def list_people():
    return vector_db.list_people()


@app.delete("/people/{person_id}")
def delete_person(person_id: int):
    deleted = vector_db.delete(person_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Persona no encontrada")
    return {"deleted": person_id}

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"

MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# --- GhostFaceNets (embedder) ---
EMBEDDER_MODEL_PATH = MODELS_DIR / "GhostFaceNet_W1.3_S2_ArcFace.h5"
EMBEDDER_MODEL_URL = (
    "https://github.com/HamadYA/GhostFaceNets/releases/download/v1.3/"
    "GhostFaceNet_W1.3_S2_ArcFace.h5"
)
EMBEDDING_SIZE = 512
FACE_SIZE = 112  # ancho/alto esperado por el embedder (112x112 RGB)

# --- Detector de rostros (YuNet, ONNX) ---
# OpenCV 5.x elimino los importadores legacy de Caffe/Darknet/Torch del modulo
# dnn, asi que usamos YuNet (el detector que el propio OpenCV recomienda ahora)
# en vez del viejo SSD res10 en formato Caffe.
DETECTOR_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
DETECTOR_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DETECTOR_CONFIDENCE_THRESHOLD = 0.6
DETECTOR_NMS_THRESHOLD = 0.3
DETECTOR_TOP_K = 5000

# --- Base vectorial (FAISS) ---
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
LABELS_PATH = DATA_DIR / "labels.json"

# Umbral de similitud coseno (producto interno de embeddings normalizados).
# Por debajo de este valor, la persona se considera "Desconocido".
MATCH_THRESHOLD = 0.4

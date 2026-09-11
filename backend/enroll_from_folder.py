"""Registra varias personas en la base vectorial a partir de una carpeta de
fotos ya existentes, sin necesidad de usar la camara en vivo.

Estructura esperada (una subcarpeta por persona, el nombre de la subcarpeta
es el nombre que se guarda):

    fotos/
        Juan/
            1.jpg
            2.jpg
        Maria/
            a.png

Uso:

    python enroll_from_folder.py ruta\a\fotos
"""

import argparse
import sys
from pathlib import Path

import cv2

from align import align_face
from embedder import Embedder
from face_detector import FaceDetector
from vector_db import VectorDB

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_person_images(root: Path):
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(
            f for f in person_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            yield person_dir.name, images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Carpeta con una subcarpeta por persona")
    args = parser.parse_args()

    root: Path = args.folder
    if not root.is_dir():
        print(f"[error] '{root}' no es una carpeta valida", file=sys.stderr)
        sys.exit(1)

    people = list(iter_person_images(root))
    if not people:
        print(f"[error] no se encontraron subcarpetas con imagenes dentro de '{root}'", file=sys.stderr)
        sys.exit(1)

    print("Cargando modelos...")
    detector = FaceDetector()
    embedder = Embedder()
    vector_db = VectorDB()

    total_added = 0
    total_skipped = 0

    for name, images in people:
        print(f"\n== {name} ({len(images)} foto(s)) ==")
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"  [skip] {image_path.name}: no se pudo leer la imagen")
                total_skipped += 1
                continue

            box = detector.detect_largest(frame)
            if box is None:
                print(f"  [skip] {image_path.name}: no se detecto ningun rostro")
                total_skipped += 1
                continue

            aligned_face = align_face(frame, box.landmarks)
            embedding = embedder.get_embedding(aligned_face)
            vector_db.add(name, embedding)
            print(f"  [ok] {image_path.name}: registrado")
            total_added += 1

    print(f"\nListo. {total_added} foto(s) agregadas, {total_skipped} omitida(s).")


if __name__ == "__main__":
    main()

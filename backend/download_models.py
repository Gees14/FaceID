"""Descarga los artefactos necesarios (pesos del embedder + detector de rostros)
a backend/models/ si todavia no existen. Se puede correr suelto:

    python download_models.py
"""

import sys

import requests

import config


def _download(url: str, dest, label: str) -> None:
    if dest.exists():
        print(f"[ok] {label} ya existe en {dest}")
        return

    print(f"[..] descargando {label} desde {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {downloaded / 1e6:.1f}MB / {total / 1e6:.1f}MB ({pct:.0f}%)", end="")
        print()
        tmp_path.replace(dest)
    print(f"[ok] {label} descargado en {dest}")


def main() -> None:
    _download(config.EMBEDDER_MODEL_URL, config.EMBEDDER_MODEL_PATH, "modelo GhostFaceNets")
    _download(config.DETECTOR_MODEL_URL, config.DETECTOR_MODEL_PATH, "detector de rostros (YuNet)")
    print("Listo. Todos los modelos estan disponibles en", config.MODELS_DIR)


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"[error] fallo la descarga: {exc}", file=sys.stderr)
        sys.exit(1)

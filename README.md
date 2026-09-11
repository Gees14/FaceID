# FaceID POC

Prueba de concepto de reconocimiento facial: detecta rostros con la cámara
de la compu en el navegador, calcula su embedding con **GhostFaceNets** en un
backend Python y lo compara contra una base vectorial local (FAISS) para
reconocer a personas previamente registradas.

## Estructura

- `backend/` — FastAPI + detector de rostros (YuNet) + embedder
  (GhostFaceNets) + base vectorial (FAISS).
- `frontend/` — HTML/CSS/JS vanilla que muestra la cámara, dibuja los
  recuadros detectados y permite registrar personas nuevas.

## Setup

1. Tener [Anaconda/Miniconda](https://docs.conda.io/) instalado.
2. Correr `setup_env.bat` desde la raíz del proyecto. Esto:
   - Crea el entorno conda `FaceID` con Python 3.10.
   - Instala las dependencias de `backend/requirements.txt`.
   - Descarga los pesos de GhostFaceNets y el detector de rostros a
     `backend/models/` (también se descargan automáticamente al primer
     arranque del backend si faltan).

## Levantar el backend

```
conda activate FaceID
cd backend
uvicorn app:app --reload --port 8000
```

## Levantar el frontend

Es HTML/CSS/JS estático, se puede servir con cualquier servidor simple
(necesario para que `getUserMedia` funcione bien en el navegador):

```
cd frontend
python -m http.server 5500
```

Y abrir `http://localhost:5500` en el navegador.

## Uso

1. Acepta el permiso de cámara.
2. Verás el video en vivo con recuadros rojos ("Desconocido") sobre los
   rostros detectados.
3. Escribe tu nombre en el panel y presiona **Registrar rostro**.
4. En los siguientes frames tu recuadro debería ponerse verde con tu nombre.
5. Puedes ver/eliminar personas registradas desde la lista del panel lateral.

## Registrar personas desde fotos existentes (sin cámara)

Si ya tienes fotos guardadas en vez de registrar en vivo, usa
`backend/enroll_from_folder.py`. Organiza una carpeta con una subcarpeta por
persona (el nombre de la subcarpeta es el nombre que se guarda):

```
fotos/
    Juan/
        1.jpg
        2.jpg
    Maria/
        a.png
```

Y corre (con el entorno `FaceID` activado, desde `backend/`):

```
python enroll_from_folder.py C:\ruta\a\fotos
```

Se puede correr varias veces y con varias fotos por persona (cada foto se
guarda como un embedding aparte, lo que ayuda a reconocerla mejor desde
distintos ángulos). Las imágenes donde no se detecte ningún rostro se
omiten con un aviso, sin interrumpir el resto del proceso.

## Notas

- El umbral de similitud para considerar un match se ajusta en
  `backend/config.py` (`MATCH_THRESHOLD`).
- `backend/models/` y `backend/data/` están en `.gitignore` (pesos pesados y
  base vectorial generada en runtime, respectivamente).

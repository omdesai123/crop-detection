"""
Crop vs Weed detector - FastAPI backend.

Loads the model bundle saved by the notebook (crop_detection_cnn.pkl) and serves:
    GET  /api/health    -> is the model loaded, which classes it knows
    POST /api/predict   -> upload one image, get the predicted class + confidence
    GET  /              -> the web interface (../frontend)

Run from the project root:
    uvicorn backend.main:app --reload
"""
import io
import json
import os
import pickle
import tempfile
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "crop_detection_cnn.pkl"))

MAX_UPLOAD_MB = 10
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
LOW_CONFIDENCE = 0.70   # below this the UI asks for a clearer photo

state = {"model": None, "class_names": [], "img_size": 224, "error": None}


def strip_unknown_keys(node, keys=("quantization_config",)):
    """Recursively remove config keys that only newer Keras versions know about."""
    if isinstance(node, dict):
        return {k: strip_unknown_keys(v, keys) for k, v in node.items() if k not in keys}
    if isinstance(node, list):
        return [strip_unknown_keys(v, keys) for v in node]
    return node


def make_compatible_copy(src: Path, dst: Path) -> None:
    """A .keras file is a zip. Copy it, removing settings an older Keras cannot read (weights untouched)."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                data = json.dumps(strip_unknown_keys(json.loads(data))).encode("utf-8")
            zout.writestr(item, data)


def load_keras_file(path: Path):
    import tensorflow as tf

    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as first_error:  # noqa: BLE001
        print(f"Normal load failed ({str(first_error).splitlines()[0][:120]}). Retrying in compatibility mode...")
        fixed = path.with_name("model_compat.keras")
        make_compatible_copy(path, fixed)
        model = tf.keras.models.load_model(fixed, compile=False)
        print("Loaded in compatibility mode (your local Keras is older than the one used for training).")
        return model


def load_model_bundle() -> None:
    """Read the pickle and rebuild the Keras model from the bytes stored inside it."""
    if not MODEL_PATH.exists():
        state["error"] = (
            f"Model file not found at {MODEL_PATH}. "
            "Copy crop_detection_cnn.pkl into the model/ folder and restart the server."
        )
        print(state["error"])
        return
    try:
        import tensorflow as tf  # imported here so the server still starts without it

        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)  # only load pickle files you created yourself
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = Path(tmp) / "model.keras"
            tmp_file.write_bytes(bundle["model_bytes"])
            state["model"] = load_keras_file(tmp_file)
        state["class_names"] = list(bundle["class_names"])
        state["img_size"] = int(bundle.get("img_size", 224))
        state["error"] = None
        # warm-up so the first real request is fast
        size = state["img_size"]
        state["model"].predict(np.zeros((1, size, size, 3), "float32"), verbose=0)
        print(f"Model loaded: classes={state['class_names']} img_size={size}")
    except Exception as exc:  # noqa: BLE001
        state["model"] = None
        state["error"] = f"Could not load the model: {exc}"
        print(state["error"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model_bundle()
    yield


app = FastAPI(title="Crop vs Weed Detector", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {
        "model_loaded": state["model"] is not None,
        "classes": state["class_names"],
        "img_size": state["img_size"],
        "max_upload_mb": MAX_UPLOAD_MB,
        "low_confidence_threshold": LOW_CONFIDENCE,
        "detail": state["error"],
    }


def preprocess(raw: bytes) -> np.ndarray:
    """Same steps as training: RGB, resize to img_size (bilinear). Scaling to 0-1 is inside the model."""
    import tensorflow as tf

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")   # fixes phone photos that are rotated
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="That file is not a readable image. Upload a JPG or PNG photo.")
    arr = np.asarray(img, dtype="float32")
    size = state["img_size"]
    return tf.image.resize(arr, (size, size)).numpy()[None, ...]


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail=state["error"] or "Model is not loaded.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a JPG or PNG image.")

    limit = MAX_UPLOAD_MB * 1024 * 1024
    raw = file.file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(status_code=413, detail=f"Image is larger than {MAX_UPLOAD_MB} MB. Upload a smaller photo.")
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    batch = preprocess(raw)
    t0 = time.perf_counter()
    probs = state["model"].predict(batch, verbose=0)[0]
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    k = int(np.argmax(probs))
    return {
        "label": state["class_names"][k],
        "confidence": float(probs[k]),
        "probabilities": {c: float(p) for c, p in zip(state["class_names"], probs)},
        "low_confidence": bool(probs[k] < LOW_CONFIDENCE),
        "inference_ms": elapsed_ms,
        "filename": file.filename,
    }


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
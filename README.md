# Fieldscan - Crop vs Weed detector

A web app for your trained CNN. FastAPI backend + plain HTML/CSS/JS frontend.

## Folder layout
```
crop-detector-app/
├── backend/
│   ├── main.py            FastAPI server (loads the .pkl, serves /api and the website)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── model/
    └── crop_detection_cnn.pkl     <- put your file here
```

## Run it
1. Copy `crop_detection_cnn.pkl` (from the Colab notebook, Cell 16b) into the `model/` folder.
2. Open a terminal in the project root (the folder that contains `backend/`):
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```
3. Open http://127.0.0.1:8000 in your browser.

Use a Python version your TensorFlow install supports (3.10 - 3.12 is a safe choice), and ideally the same TensorFlow version you trained with in Colab (`import tensorflow as tf; tf.__version__`).

## API
| Method | URL | What it does |
|---|---|---|
| GET | `/api/health` | Model status, class names, image size |
| POST | `/api/predict` | Form field `file` (JPG/PNG, max 10 MB). Returns label, confidence, per-class probabilities |
| GET | `/docs` | Automatic interactive API docs |

Example:
```bash
curl -F "file=@plant.jpg" http://127.0.0.1:8000/api/predict
```

## Settings
* Different pickle location: set `MODEL_PATH`, e.g. `MODEL_PATH=D:/models/crop.pkl uvicorn backend.main:app`
* Low-confidence warning level: `LOW_CONFIDENCE` in `backend/main.py` (default 0.70)
* Max upload size: `MAX_UPLOAD_MB` in `backend/main.py`

## Troubleshooting
* **"Model file not found"**: the pickle is not in `model/` or the name differs.
* **"Could not load the model"**: usually a TensorFlow/Keras version mismatch. Install the Colab version.
* **"Server offline"** in the page: the server is not running, or you opened `index.html` directly as a file. Always open the address from step 3.

Only load pickle files you created yourself: unpickling a file from someone else can run code.

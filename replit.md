# TrueSource AI

TrueSource AI is a web app that detects whether an image is real or AI-generated, using a Keras deep-learning model. Users upload an image or capture one with their camera and get a confidence verdict (green "Real" / red "AI-Generated").

## Run & Operate

- The app runs via the `artifacts/truesource-ai: web` workflow (gunicorn serving the Flask app at `/`).
- App code lives in `truesource/` (Python Flask). The `artifacts/truesource-ai/` directory holds the artifact registration; its `artifact.toml` points the dev + production run commands at the Flask app.
- Dev/prod run command: `cd /home/runner/workspace/truesource && /home/runner/workspace/.pythonlibs/bin/gunicorn --workers 1 --threads 8 --timeout 180 --bind 0.0.0.0:$PORT app:app`
- Server binds `0.0.0.0` and reads the `PORT` env var (gunicorn `--bind`).
- Python deps are in `truesource/requirements.txt` and installed into `.pythonlibs/`.

## Stack

- Backend: Python 3.11, Flask, served by gunicorn (1 worker, 8 threads).
- Model: Keras `deepfake_detector.keras`, auto-downloaded from Hugging Face Hub repo `GorangSaini/truesource-ai-model` on first run.
- ML runtime: tensorflow-cpu, numpy, Pillow, huggingface_hub.
- Frontend: vanilla HTML/CSS/JS (no framework) — `truesource/templates/index.html`, `truesource/static/style.css`, `truesource/static/app.js`.

## Where things live

- `truesource/app.py` — Flask app: loads the model in a background thread, serves the UI and API.
- `truesource/helpers.py` — model contract: preprocess to 128x128 RGB, normalize /255, single sigmoid = P(AI-Generated); label AI if prob >= 0.5. Classes: `["Real", "AI-Generated"]`.
- `truesource/model_setup.py`, `truesource/model_config.json`, `truesource/class_names.json` — model paths + HF Hub config.
- `truesource/sample_images/` — 5 Real + 5 AI-Generated sample PNGs.
- `truesource/static/app.js` — `BASE = window.location.pathname`; all API calls are relative so the app works behind the proxy path prefix.

## API endpoints

Endpoints deliberately avoid the `/api` prefix (the scaffold `api-server` artifact claims `/api`):

- `GET /healthz` — `{ ready, error }` model-load status (polled by the UI).
- `GET /samples-list` — list of sample images with their true labels.
- `GET /samples/<file>` — serve a sample image.
- `POST /predict` — accepts a file upload (multipart), `{ "sample": name }`, or `{ "image_data": base64 }`; returns `{ label, confidence, prob_real, prob_ai }`.

## Architecture decisions

- Built as a standalone Flask app (not the Node `api-server`) because the model is Python/Keras. Registered as a `kind="web"` artifact at `/` so it shows in the preview pane and deploys; the artifact's `artifact.toml` was rewritten to launch gunicorn for both dev and production instead of the default Vite/static hosting.
- The React scaffold files under `artifacts/truesource-ai/src/` are unused — the artifact only exists to register the web service and route `/` to the Flask app.
- The model loads in a background thread at import so the server binds the port immediately; the UI polls `/healthz` and shows a loading state until ready. First run downloads the model from HF Hub.
- `huggingface_hub` must be installed via plain pip (`.pythonlibs/bin/python -m pip install huggingface_hub`); installing via uv forced a pytorch-cpu index and failed.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- The `artifact.toml` run command's working directory differs between dev (artifact dir) and production (repo root), so the run commands use absolute paths.
- The scaffold `api-server` (at `/api`) and `mockup-sandbox` (Canvas, at `/__mockup`) artifacts are unused by this product but left in place; keep TrueSource's endpoints off the `/api` prefix to avoid the api-server intercepting them.
- `TF_CPP_MIN_LOG_LEVEL=3` is set to quiet TensorFlow startup logs.

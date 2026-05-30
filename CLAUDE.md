# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FaceMakeIt — automated photo editing that detects and removes background faces (posters, reflections, passersby) while keeping the main subject. Frontend is plain HTML + Tailwind CSS served as static files by FastAPI. Backend is Python FastAPI with an ML pipeline (InsightFace face detection → scoring → smart masking → Stable Diffusion inpainting on GPU).

## Commands

```bash
# Run locally (from project root)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run via Docker (requires NVIDIA GPU)
docker compose up --build

# Install dependencies
pip install -r requirements.txt

# Access
# App:      http://localhost:8000
# Swagger:  http://localhost:8000/docs
# ReDoc:    http://localhost:8000/redoc
```

## Architecture

```
app/
├── main.py          # FastAPI routes, auth, response models, static mounts
├── config.py        # Settings from .env, paths, constants
├── pipeline.py      # Orchestrates: detect → score → mask → inpaint
├── detection.py     # InsightFace face detection (lazy-loaded model)
├── scoring.py       # Main subject selection (size 50%, center 30%, quality 20%)
├── masking.py       # Smart body detection — full body mask vs face-only mask
├── inpainting.py    # Stable Diffusion inpainting (GPU, lazy-loaded pipeline)
└── static/          # Frontend (plain HTML + Tailwind CDN + vanilla JS)
```

**ML Pipeline** (`pipeline.py`): Runs async via FastAPI BackgroundTasks. Status tracked in `data/uploads/{task_id}/status.json`. Frontend polls `/api/status/{task_id}` every 500ms.

**Smart Masking** (`masking.py`): Analyzes region below each background face to decide full body removal (texture/edge/color analysis) vs face-only removal (for posters/frames).

**Inpainting** (`inpainting.py`): Resizes to 512x512 for SD, runs inpainting on GPU, then blends result back into original using feathered mask edges. Uses `runwayml/stable-diffusion-inpainting` via `diffusers`.

## API Response Format

All JSON responses use a consistent envelope:
```json
{"success": true, "data": {...}, "error": null}   // success
{"success": false, "data": null, "error": "..."}    // error
```

File download endpoints (`/api/result/{id}`, `/api/mask/{id}`) return binary data directly.

## Auth

Bearer token auth via `API_KEY` env var. If `API_KEY` is empty, auth is disabled (open access). Frontend stores key in localStorage.

## Imports

All imports use `app.` prefix: `from app.config import ...`, `from app.pipeline import ...`. Run uvicorn from project root, not from inside `app/`.

## Data

- `data/uploads/{task_id}/` — uploaded files + status.json + faces.json
- `data/results/{task_id}/` — result.jpg + mask.png
- `data/samples/` — test images
- `data/` is gitignored

## User Guidelines (from RULE.md)

Be direct and honest. Don't agree by default — challenge weak ideas and point out flaws. Be concise. No empty praise. Focus on what's useful.

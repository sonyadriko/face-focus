"""FaceMakeIt - FastAPI application."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import UPLOAD_DIR, RESULTS_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS, API_KEY
from app.pipeline import run_pipeline


# --- Auth ---

security = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not API_KEY:
        return
    if not credentials or credentials.credentials != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")


# --- Response models ---

class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


class UploadData(BaseModel):
    task_id: str
    filename: str
    created_at: str


class ProcessData(BaseModel):
    task_id: str
    status: str


class StatusData(BaseModel):
    task_id: str
    status: str
    stage: str
    progress: int
    filename: Optional[str] = None
    extension: Optional[str] = None
    faces_detected: Optional[int] = None
    faces_removed: Optional[int] = None
    main_face_bbox: Optional[list] = None
    message: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None


class FaceInfo(BaseModel):
    bbox: list[float]
    confidence: float
    score: float
    is_main: bool


class HealthData(BaseModel):
    status: str
    version: str


# --- App ---

app = FastAPI(
    title="FaceMakeIt API",
    version="1.0.0",
    description="Automated photo editing — detect and remove background faces (posters, reflections, passersby) while keeping the main subject.",
    docs_url="/docs",
    redoc_url="/redoc",
    dependencies=[Depends(verify_token)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "data": None, "error": exc.detail},
    )


# --- Helpers ---

def ok(data: Any) -> dict:
    return {"success": True, "data": data, "error": None}


def _update_status(task_dir: Path, **kwargs):
    path = task_dir / "status.json"
    data = json.loads(path.read_text())
    data.update(kwargs)
    path.write_text(json.dumps(data))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Endpoints ---

@app.get("/api/health", response_model=ApiResponse, tags=["System"])
def health():
    """Health check."""
    return ok(HealthData(status="ok", version="1.0.0"))


@app.post("/api/upload", response_model=ApiResponse, tags=["Pipeline"],
          responses={400: {"model": ApiResponse}, 401: {"model": ApiResponse}})
async def upload_image(file: UploadFile = File(..., description="Image file (JPG, PNG, WebP, max 20MB)")):
    """Upload an image for processing.

    Accepts JPG, PNG, or WebP files up to 20MB.
    Returns a task_id used in subsequent pipeline steps.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large (max 20MB)")

    task_id = str(uuid.uuid4())
    now = _now_iso()
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True)
    (task_dir / f"original{ext}").write_bytes(content)

    (task_dir / "status.json").write_text(json.dumps({
        "task_id": task_id, "status": "uploaded", "stage": "uploaded",
        "progress": 0, "filename": file.filename, "extension": ext,
        "created_at": now,
    }))

    return ok(UploadData(task_id=task_id, filename=file.filename, created_at=now))


@app.post("/api/process/{task_id}", response_model=ApiResponse, tags=["Pipeline"],
          responses={404: {"model": ApiResponse}})
async def process_image(task_id: str, background_tasks: BackgroundTasks):
    """Start the face removal pipeline.

    Runs asynchronously in the background. Poll /api/status/{task_id} for progress.
    Pipeline stages: detecting → scoring → masking → inpainting → done.
    """
    task_dir = UPLOAD_DIR / task_id
    if not task_dir.exists():
        raise HTTPException(404, "Task not found")

    _update_status(task_dir, status="processing", stage="starting", progress=5)
    background_tasks.add_task(run_pipeline, task_id)
    return ok(ProcessData(task_id=task_id, status="processing"))


@app.get("/api/status/{task_id}", response_model=ApiResponse, tags=["Pipeline"],
         responses={404: {"model": ApiResponse}})
def get_status(task_id: str):
    """Get processing status and progress.

    Poll every 500ms while status is "processing".
    When done, call /api/result/{task_id} to download.
    """
    path = UPLOAD_DIR / task_id / "status.json"
    if not path.exists():
        raise HTTPException(404, "Task not found")
    return ok(StatusData(**json.loads(path.read_text())))


@app.get("/api/faces/{task_id}", response_model=ApiResponse, tags=["Results"],
         responses={404: {"model": ApiResponse}})
def get_faces(task_id: str):
    """Get face detection results.

    Returns all detected faces with bounding boxes, scores, and main subject flag.
    """
    path = UPLOAD_DIR / task_id / "faces.json"
    if not path.exists():
        raise HTTPException(404, "Face data not found. Process the image first.")
    return ok([FaceInfo(**f) for f in json.loads(path.read_text())])


@app.get("/api/result/{task_id}", tags=["Results"],
         responses={404: {"model": ApiResponse}, 200: {"content": {"image/jpeg": {}}}})
def get_result(task_id: str):
    """Download the processed image.

    Returns the edited image with background faces removed.
    Only available after processing status is "done".
    """
    path = RESULTS_DIR / task_id / "result.jpg"
    if not path.exists():
        raise HTTPException(404, "Result not ready yet.")
    return FileResponse(path, media_type="image/jpeg", filename=f"facemakeit_{task_id}.jpg")


@app.get("/api/mask/{task_id}", tags=["Results"],
         responses={404: {"model": ApiResponse}, 200: {"content": {"image/png": {}}}})
def get_mask(task_id: str):
    """Download the removal mask.

    Returns the binary mask used for inpainting (white = removed area).
    Useful for debugging.
    """
    path = RESULTS_DIR / task_id / "mask.png"
    if not path.exists():
        raise HTTPException(404, "Mask not found.")
    return FileResponse(path, media_type="image/png")


# Static file mounts
STATIC_DIR = Path(__file__).parent / "static"

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

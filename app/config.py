"""FaceMakeIt configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.parent
API_KEY = os.getenv("API_KEY", "")
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

FACE_CONFIDENCE_THRESHOLD = 0.5

SCORE_WEIGHT_SIZE = 0.50
SCORE_WEIGHT_CENTER = 0.30
SCORE_WEIGHT_QUALITY = 0.20

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

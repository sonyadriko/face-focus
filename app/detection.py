"""Face detection using InsightFace."""
import cv2
from app.config import FACE_CONFIDENCE_THRESHOLD

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from insightface.app import FaceAnalysis
        _analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return _analyzer


def detect_faces(image_path: str) -> list[dict]:
    """Detect all faces in an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    faces = _get_analyzer().get(img)
    return [
        {
            "bbox": face.bbox.tolist(),
            "confidence": round(float(face.det_score), 4),
        }
        for face in faces
        if face.det_score >= FACE_CONFIDENCE_THRESHOLD
    ]

"""ML pipeline: detect → score → mask → inpaint."""
import json
import cv2
from pathlib import Path

from app.config import UPLOAD_DIR, RESULTS_DIR
from app.detection import detect_faces
from app.scoring import score_faces
from app.masking import generate_mask
from app.inpainting import inpaint


def _update_status(task_dir: Path, **kwargs):
    path = task_dir / "status.json"
    data = json.loads(path.read_text())
    data.update(kwargs)
    path.write_text(json.dumps(data))


def run_pipeline(task_id: str):
    """Run the full face removal pipeline."""
    task_dir = UPLOAD_DIR / task_id
    result_dir = RESULTS_DIR / task_id
    result_dir.mkdir(parents=True, exist_ok=True)

    status = json.loads((task_dir / "status.json").read_text())
    ext = status.get("extension", ".jpg")
    original = str(task_dir / f"original{ext}")

    try:
        # Detect
        _update_status(task_dir, stage="detecting", progress=15, status="processing")
        faces = detect_faces(original)
        _update_status(task_dir, faces_detected=len(faces), progress=35)

        if not faces:
            _update_status(task_dir, stage="no_faces", progress=100, status="done",
                           faces_removed=0, message="No faces detected.")
            cv2.imwrite(str(result_dir / "result.jpg"), cv2.imread(original))
            return

        # Score
        _update_status(task_dir, stage="scoring", progress=45)
        img = cv2.imread(original)
        h, w = img.shape[:2]
        scored = score_faces(faces, w, h)
        (task_dir / "faces.json").write_text(json.dumps(scored, indent=2))

        bg = [f for f in scored if not f["is_main"]]
        main = next((f for f in scored if f["is_main"]), None)
        _update_status(task_dir, faces_removed=len(bg), progress=55,
                       main_face_bbox=main["bbox"] if main else None)

        if not bg:
            _update_status(task_dir, stage="no_background", progress=100, status="done",
                           message="Only one face. No removal needed.")
            cv2.imwrite(str(result_dir / "result.jpg"), img)
            return

        # Mask
        _update_status(task_dir, stage="masking", progress=65)
        mask = generate_mask(scored, w, h, image=img)
        mask_path = str(result_dir / "mask.png")
        cv2.imwrite(mask_path, mask)

        # Inpaint
        _update_status(task_dir, stage="inpainting", progress=75)
        inpaint(original, mask_path, str(result_dir / "result.jpg"))

        _update_status(task_dir, stage="done", progress=100, status="done",
                       message="Processing complete.")

    except Exception as e:
        _update_status(task_dir, stage="error", status="error",
                       error=str(e), message=f"Error: {e}")

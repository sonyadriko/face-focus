"""ML pipeline: detect → score → mask → inpaint."""
import json
import cv2
from pathlib import Path

from app.config import UPLOAD_DIR, RESULTS_DIR
from app.detection import detect_faces
from app.scoring import score_faces
from app.masking import generate_mask, generate_blur_mask, apply_blur
from app.inpainting import inpaint


def _update_status(task_dir: Path, **kwargs):
    path = task_dir / "status.json"
    data = json.loads(path.read_text())
    data.update(kwargs)
    path.write_text(json.dumps(data))


def run_pipeline(task_id: str, mode: str = "remove"):
    """Run the face processing pipeline.

    Args:
        task_id: task identifier
        mode: "remove" (inpaint) or "blur" (Gaussian blur)
    """
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

        # Mask + Process
        if mode == "blur":
            _update_status(task_dir, stage="masking", progress=65)
            mask = generate_blur_mask(scored, w, h)
            _update_status(task_dir, stage="blurring", progress=75)
            result = apply_blur(img, mask)
            cv2.imwrite(str(result_dir / "result.jpg"), result)
        else:
            _update_status(task_dir, stage="masking", progress=65)
            mask = generate_mask(scored, w, h, image=img)
            mask_path = str(result_dir / "mask.png")
            cv2.imwrite(mask_path, mask)

            # Process each face individually for better quality
            _update_status(task_dir, stage="inpainting", progress=70)
            result_path = str(result_dir / "result.jpg")
            current_img = img.copy()
            bg_faces = [f for f in scored if not f["is_main"]]

            for i, face in enumerate(bg_faces):
                # Generate mask for this single face
                single_mask = generate_mask([face, *[f for f in scored if f["is_main"]]],
                                            w, h, image=current_img)
                single_mask_path = str(result_dir / "_temp_mask.png")
                cv2.imwrite(single_mask_path, single_mask)

                # Save current state for inpainting
                temp_input = str(result_dir / "_temp_input.jpg")
                cv2.imwrite(temp_input, current_img)

                # Inpaint this face
                inpaint(temp_input, single_mask_path, result_path)

                # Load result for next iteration
                current_img = cv2.imread(result_path)

                # Update progress
                pct = 70 + int((i + 1) / len(bg_faces) * 25)
                _update_status(task_dir, progress=pct)

            # Cleanup temp files
            (result_dir / "_temp_mask.png").unlink(missing_ok=True)
            (result_dir / "_temp_input.jpg").unlink(missing_ok=True)

        _update_status(task_dir, stage="done", progress=100, status="done",
                       message="Processing complete.")

    except Exception as e:
        _update_status(task_dir, stage="error", status="error",
                       error=str(e), message=f"Error: {e}")

"""Main subject face scoring."""
from app.config import SCORE_WEIGHT_SIZE, SCORE_WEIGHT_CENTER, SCORE_WEIGHT_QUALITY


def score_faces(faces: list[dict], image_width: int, image_height: int) -> list[dict]:
    """Score each face by size, center proximity, and detection quality.

    Returns faces with 'score' and 'is_main' fields added.
    """
    if not faces:
        return []

    img_area = image_width * image_height
    cx, cy = image_width / 2, image_height / 2
    diagonal = (image_width ** 2 + image_height ** 2) ** 0.5

    scored = []
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        face_area = (x2 - x1) * (y2 - y1)

        size = min(face_area / (img_area * 0.15), 1.0)
        dist = (((x1 + x2) / 2 - cx) ** 2 + ((y1 + y2) / 2 - cy) ** 2) ** 0.5
        center = max(0, 1 - dist / (diagonal / 2))
        quality = face["confidence"]

        total = (size * SCORE_WEIGHT_SIZE) + (center * SCORE_WEIGHT_CENTER) + (quality * SCORE_WEIGHT_QUALITY)

        scored.append({
            **face,
            "score": round(total, 4),
            "is_main": False,
        })

    max_idx = max(range(len(scored)), key=lambda i: scored[i]["score"])
    scored[max_idx]["is_main"] = True

    return scored

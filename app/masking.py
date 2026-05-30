"""Mask generation with smart body detection.

Analyzes the region below each background face:
- Body detected (texture, edges, skin tones) → full person mask
- No body (poster, frame, reflection) → face-only mask
"""
import cv2
import numpy as np

VARIANCE_THRESHOLD = 300
EDGE_DENSITY_THRESHOLD = 0.05


def _has_body_below(image: np.ndarray, bbox: list) -> bool:
    """Check if there's a body below the face by analyzing texture, edges, and color diversity."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    face_h, face_w = y2 - y1, x2 - x1

    # Sample region: below face, 3x face height, 2x face width
    r_top, r_bot = y2, min(h, y2 + face_h * 3)
    r_left, r_right = max(0, x1 - face_w), min(w, x2 + face_w)

    if r_bot <= r_top or r_right <= r_left:
        return False

    region = image[r_top:r_bot, r_left:r_right]
    if region.size == 0:
        return False

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Variance: flat = poster, textured = person
    has_texture = gray.var() > VARIANCE_THRESHOLD
    # Edges: bodies have clothing folds, limbs
    has_edges = np.count_nonzero(cv2.Canny(gray, 50, 150)) / gray.size > EDGE_DENSITY_THRESHOLD
    # Color diversity: bodies have multiple colors (skin + clothing)
    has_color = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[:, :, 0].std() > 15

    return sum([has_texture, has_edges, has_color]) >= 2


def _face_mask(mask: np.ndarray, bbox: list, w: int, h: int):
    """Face-only mask with padding (for posters/frames)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fw, fh = x2 - x1, y2 - y1
    pad_x, pad_y = int(fw * 0.5), int(fh * 0.5)

    mx1, my1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    mx2, my2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

    cx, cy = (mx1 + mx2) // 2, (my1 + my2) // 2
    cv2.ellipse(mask, (cx, cy), ((mx2 - mx1) // 2, (my2 - my1) // 2), 0, 0, 360, 255, -1)


def _body_mask(mask: np.ndarray, bbox: list, w: int, h: int):
    """Full person mask: head → shoulders → torso → legs → feet."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fw, fh = x2 - x1, y2 - y1
    fcx = (x1 + x2) / 2

    shoulder_w = fw * 3.0
    feet_w = fw * 3.5

    top = max(0, int(y1 - fh * 0.7))
    shoulder = int(y2 + fh * 1.0)
    waist = int(y2 + fh * 3.0)
    knee = int(y2 + fh * 4.5)

    ls = max(0, int(fcx - shoulder_w / 2))
    rs = min(w, int(fcx + shoulder_w / 2))
    lw = max(0, int(fcx - shoulder_w * 0.4))
    rw = min(w, int(fcx + shoulder_w * 0.4))
    lf = max(0, int(fcx - feet_w / 2))
    rf = min(w, int(fcx + feet_w / 2))

    pts = np.array([
        [int(fcx), top], [ls, shoulder], [lw, waist],
        [lf, knee], [lf, h], [rf, h],
        [rf, knee], [rw, waist], [rs, shoulder],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # Rounded head
    head_cy = (top + shoulder) // 2
    cv2.ellipse(mask, (int(fcx), head_cy), (int(shoulder_w / 2), (shoulder - top) // 2), 0, 0, 360, 255, -1)


def generate_mask(faces: list[dict], image_width: int, image_height: int,
                   image: np.ndarray = None) -> np.ndarray:
    """Generate smart mask for background faces.

    For each background face, analyzes the region below to decide
    whether to mask just the face or the full body.
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for face in faces:
        if face.get("is_main"):
            continue

        if image is not None and _has_body_below(image, face["bbox"]):
            _body_mask(mask, face["bbox"], image_width, image_height)
        else:
            _face_mask(mask, face["bbox"], image_width, image_height)

    # Smooth edges for better inpainting
    if np.any(mask > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (31, 31), 15)

    return mask

"""Mask generation with smart face type detection.

Detects whether a background face is from a poster/frame (uniform background)
or a real person (complex background), and applies appropriate mask size.
"""
import cv2
import numpy as np

# Thresholds
BG_UNIFORMITY_THRESHOLD = 400    # low variance = uniform bg (poster)
BG_EDGE_THRESHOLD = 0.03         # low edges = flat surface (poster)


def _is_poster_face(image: np.ndarray, bbox: list) -> bool:
    """Check if a face is on a poster/frame by analyzing the SURROUNDING background.

    Poster/frame faces have uniform background around them (wall, sky, flat surface).
    Real person faces have complex background (objects, mixed colors, depth).

    Returns True if the face appears to be on a poster/frame.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fh, fw = y2 - y1, x2 - x1

    # Sample region: ring around the face (not below, but AROUND)
    # Expand bbox by 1.5x in each direction, then subtract the face itself
    pad = max(fh, fw) * 1.5
    rx1 = max(0, int(x1 - pad))
    ry1 = max(0, int(y1 - pad))
    rx2 = min(w, int(x2 + pad))
    ry2 = min(h, int(y2 + pad))

    # Create mask for surrounding area (exclude face bbox)
    region_mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=np.uint8) * 255
    # Zero out the face area within the region
    fx1, fy1 = x1 - rx1, y1 - ry1
    fx2, fy2 = x2 - rx1, y2 - ry1
    fx1, fy1 = max(0, fx1), max(0, fy1)
    fx2, fy2 = min(region_mask.shape[1], fx2), min(region_mask.shape[0], fy2)
    region_mask[fy1:fy2, fx1:fx2] = 0

    region = image[ry1:ry2, rx1:rx2]
    if region.size == 0:
        return True  # assume poster if can't analyze

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Apply mask to only analyze surrounding pixels
    masked_gray = cv2.bitwise_and(gray, gray, mask=region_mask)
    pixels = masked_gray[region_mask > 0]
    if len(pixels) == 0:
        return True

    # Uniformity: low variance = flat background = poster
    bg_variance = pixels.var()
    is_uniform = bg_variance < BG_UNIFORMITY_THRESHOLD

    # Edge density: few edges = flat surface = poster
    edges = cv2.Canny(gray, 50, 150)
    masked_edges = cv2.bitwise_and(edges, edges, mask=region_mask)
    edge_density = np.count_nonzero(masked_edges) / max(np.count_nonzero(region_mask), 1)
    is_flat = edge_density < BG_EDGE_THRESHOLD

    # If background is uniform AND flat → poster
    return is_uniform and is_flat


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

    head_cy = (top + shoulder) // 2
    cv2.ellipse(mask, (int(fcx), head_cy), (int(shoulder_w / 2), (shoulder - top) // 2), 0, 0, 360, 255, -1)


def _main_subject_protection(faces: list[dict], w: int, h: int) -> np.ndarray:
    """Create protection mask for main subject's FACE only.

    Only protects the face area with moderate padding — not the entire body.
    This way background person masks can still overlap the main subject's
    body area, but the main subject's face is never inpainted.
    """
    protection = np.zeros((h, w), dtype=np.uint8)

    for face in faces:
        if not face.get("is_main"):
            continue

        x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
        fh, fw = y2 - y1, x2 - x1

        # Protect face + moderate padding only
        pad_x = int(fw * 0.8)
        pad_y = int(fh * 0.8)

        mx1 = max(0, x1 - pad_x)
        my1 = max(0, y1 - pad_y)
        mx2 = min(w, x2 + pad_x)
        my2 = min(h, y2 + pad_y)

        cx, cy = (mx1 + mx2) // 2, (my1 + my2) // 2
        cv2.ellipse(protection, (cx, cy), ((mx2 - mx1) // 2, (my2 - my1) // 2),
                     0, 0, 360, 255, -1)

    return protection


def generate_mask(faces: list[dict], image_width: int, image_height: int,
                   image: np.ndarray = None) -> np.ndarray:
    """Generate smart mask for background faces.

    For each background face, analyzes the surrounding background:
    - Uniform background (poster/frame) → face-only mask
    - Complex background (real person) → full body mask

    Main subject area is always protected from the mask.
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for face in faces:
        if face.get("is_main"):
            continue

        if image is not None and _is_poster_face(image, face["bbox"]):
            _face_mask(mask, face["bbox"], image_width, image_height)
        else:
            _body_mask(mask, face["bbox"], image_width, image_height)

    # Protect main subject area
    protection = _main_subject_protection(faces, image_width, image_height)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(protection))

    if np.any(mask > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (31, 31), 15)

    return mask


def generate_blur_mask(faces: list[dict], image_width: int, image_height: int,
                        padding_ratio: float = 0.4) -> np.ndarray:
    """Generate mask for blur mode — face bbox with padding only."""
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for face in faces:
        if face.get("is_main"):
            continue

        x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
        fw, fh = x2 - x1, y2 - y1
        pad_x = int(fw * padding_ratio)
        pad_y = int(fh * padding_ratio)

        mx1 = max(0, x1 - pad_x)
        my1 = max(0, y1 - pad_y)
        mx2 = min(image_width, x2 + pad_x)
        my2 = min(image_height, y2 + pad_y)

        cx, cy = (mx1 + mx2) // 2, (my1 + my2) // 2
        cv2.ellipse(mask, (cx, cy), ((mx2 - mx1) // 2, (my2 - my1) // 2), 0, 0, 360, 255, -1)

    if np.any(mask > 0):
        mask = cv2.GaussianBlur(mask, (21, 21), 10)

    return mask


def apply_blur(image: np.ndarray, mask: np.ndarray, blur_strength: int = 99) -> np.ndarray:
    """Apply Gaussian blur to masked areas of the image."""
    if blur_strength % 2 == 0:
        blur_strength += 1

    blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    mask_3ch = np.stack([mask] * 3, axis=-1).astype(np.float32) / 255.0
    mask_3ch = cv2.GaussianBlur(mask_3ch, (11, 11), 5)

    return (blurred * mask_3ch + image * (1 - mask_3ch)).astype(np.uint8)

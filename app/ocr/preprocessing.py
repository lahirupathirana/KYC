"""
Image preprocessing and quality validation for ID document scanning.

Pipeline (in order):
  decode → validate_quality → preprocess

preprocess steps:
  1. Perspective correction — detect document quad, apply homography warp
  2. Upscale to 1400 px wide
  3. CLAHE on LAB L-channel (local contrast)
  4. Fast colour denoise (conservative)
  5. Deskew via Hough line detection
  6. Unsharp mask

All functions are pure (no side effects, no model calls) and run synchronously
inside a ThreadPoolExecutor via the service layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# ── Quality thresholds ────────────────────────────────────────────────────────
_MIN_W = 500
_MIN_H = 320
_BLUR_THRESHOLD = 60.0      # Laplacian variance; below this → blurry
_BRIGHTNESS_MIN = 35.0      # mean gray intensity (0-255)
_BRIGHTNESS_MAX = 220.0
_CONTRAST_MIN = 20.0        # gray std-dev; below this → washed out / flat

# Perspective correction: only warp if the document quad covers at least
# this fraction of the image area (avoids warping on non-document images).
_MIN_DOC_AREA_RATIO = 0.15


@dataclass
class QualityReport:
    passed: bool
    resolution_ok: bool
    sharpness: float
    brightness: float
    contrast: float
    issues: list[str] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def decode_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupt data")
    return img


def validate_quality(img: np.ndarray) -> QualityReport:
    """
    Fast quality gate run BEFORE preprocessing.
    Returns a report rather than raising so the caller can decide whether
    to abort or attempt OCR anyway (useful for research evaluation).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    issues: list[str] = []

    resolution_ok = w >= _MIN_W and h >= _MIN_H
    if not resolution_ok:
        issues.append(f"Resolution {w}×{h} below minimum {_MIN_W}×{_MIN_H}")

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < _BLUR_THRESHOLD:
        issues.append(f"Image blurry (sharpness={sharpness:.1f}, min={_BLUR_THRESHOLD})")

    brightness = float(gray.mean())
    if brightness < _BRIGHTNESS_MIN:
        issues.append(f"Too dark (brightness={brightness:.1f})")
    elif brightness > _BRIGHTNESS_MAX:
        issues.append(f"Overexposed (brightness={brightness:.1f})")

    contrast = float(gray.std())
    if contrast < _CONTRAST_MIN:
        issues.append(f"Low contrast (contrast={contrast:.1f}, min={_CONTRAST_MIN})")

    return QualityReport(
        passed=len(issues) == 0,
        resolution_ok=resolution_ok,
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        issues=issues,
    )


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Enhancement pipeline optimised for printed ID document text.

    Steps:
      1. Perspective correction — detect document boundary quad and warp to
         a flat rectangle; critical for mobile photos taken at an angle.
      2. Upscale to at least 1400 px wide (PaddleOCR text detection works better
         at higher resolution for small fonts like MRZ lines).
      3. CLAHE on the L channel (LAB space) — improves local contrast without
         blowing highlights; better than global histogram equalisation.
      4. Fast colour denoise — reduces JPEG artefacts and camera noise while
         preserving sharp text edges (low h/hColor values).
      5. Deskew — correct residual rotation using Hough line detection.
      6. Unsharp mask — enhance edge crispness for OCR character segmentation.
    """
    img = _correct_perspective(img)
    img = _upscale(img, target_width=1400)
    img = _apply_clahe(img)
    img = _denoise(img)
    img = _deskew(img)
    img = _sharpen(img)
    return img


# ── Private helpers ───────────────────────────────────────────────────────────

def _correct_perspective(img: np.ndarray) -> np.ndarray:
    """
    Detect the document's four corners and warp to a flat rectangle.

    Works well for mobile photos where the card or passport is lying on a
    contrasting background (table, dark surface). Falls back to the original
    image when no clear quad is found.

    Algorithm:
      1. Grayscale → Gaussian blur → Canny edges
      2. Find contours sorted by area (largest first)
      3. Approximate each contour; accept the first 4-vertex polygon that
         covers at least _MIN_DOC_AREA_RATIO of the image area
      4. Order corners (TL, TR, BR, BL) → four_point_transform
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = h * w
    doc_quad: np.ndarray | None = None

    for cnt in contours[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quad_area = cv2.contourArea(approx)
            if quad_area / img_area >= _MIN_DOC_AREA_RATIO:
                doc_quad = approx.reshape(4, 2).astype(np.float32)
                break

    if doc_quad is None:
        return img

    pts = _order_quad_points(doc_quad)
    return _four_point_transform(img, pts)


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Sort four (x, y) points into [TL, TR, BR, BL] order.

    Top-left has the smallest sum; bottom-right has the largest sum.
    Top-right has the smallest difference; bottom-left has the largest difference.
    """
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]    # TL
    ordered[2] = pts[np.argmax(sums)]    # BR
    ordered[1] = pts[np.argmin(diffs)]   # TR
    ordered[3] = pts[np.argmax(diffs)]   # BL
    return ordered


def _four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a perspective warp given ordered corner points [TL, TR, BR, BL]."""
    tl, tr, br, bl = pts

    # Compute output width: max of top and bottom edge lengths
    width_top = float(np.linalg.norm(tr - tl))
    width_bot = float(np.linalg.norm(br - bl))
    out_w = max(int(width_top), int(width_bot))

    # Compute output height: max of left and right edge lengths
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    out_h = max(int(height_left), int(height_right))

    if out_w <= 0 or out_h <= 0:
        return img

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC)


def _upscale(img: np.ndarray, target_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= target_width:
        return img
    scale = target_width / w
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _denoise(img: np.ndarray) -> np.ndarray:
    # h=5, hColor=5: conservative — preserves text edges
    return cv2.fastNlMeansDenoisingColored(
        img, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21
    )


def _deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10
    )
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angles.append(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))

    # Keep only near-horizontal lines (ID cards are landscape; text lines are ~0°)
    angles = [a for a in angles if abs(a) < 25]
    if not angles:
        return img

    angle = float(np.median(angles))
    if abs(angle) < 0.8:   # skip trivial rotation
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _sharpen(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    return cv2.addWeighted(img, 1.4, blur, -0.4, 0)

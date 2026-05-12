"""
Light image preparation for face recognition.

InsightFace (buffalo_l) was trained on naturalistic RGB images.
Heavy filters hurt embedding quality, so we apply only two treatments:
  1. Gamma correction for severe under/over-exposure.
  2. Unsharp masking as a fallback when sharpness fails the quality check —
     enhances edges without altering the global texture ArcFace relies on.
"""

from __future__ import annotations

import numpy as np


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes → BGR uint8 ndarray. Raises ValueError on failure."""
    import cv2

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupt data")
    return img


def maybe_correct_gamma(img: np.ndarray) -> np.ndarray:
    """
    Apply gamma correction only when the image is unusually dark or bright.

    Leaves well-exposed images untouched to preserve the natural texture that
    ArcFace embeddings depend on.
    """
    gray = img.mean()
    if gray < 50:          # very dark — lighten
        gamma = 0.5
    elif gray > 210:       # very bright — darken
        gamma = 2.0
    else:
        return img

    import cv2

    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(img, table)


def enhance_for_recognition(img: np.ndarray) -> np.ndarray:
    """
    Two-pass enhancement for blurry document / camera-photographed ID faces.

    Pass 1 — denoise: bilateral filter preserves edges while removing camera
              noise that makes the Laplacian variance misleadingly low.
    Pass 2 — unsharp mask: sharpens edges using a wide Gaussian so fine
              texture (skin, hair) is enhanced without ringing artefacts.

    Applied automatically in pipeline.py only when sharpness is the sole
    quality failure — never applied unconditionally.
    """
    import cv2

    # Bilateral filter: smooth noise while keeping face edges crisp
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Unsharp mask on the denoised result
    blurred = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    sharpened = cv2.addWeighted(denoised, 1.8, blurred, -0.8, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

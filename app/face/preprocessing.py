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
    Unsharp masking — sharpens edges in blurry document / camera photos.

    Applied automatically in pipeline.py when the initial sharpness check
    fails but everything else passes (face detected, size ok, pose ok).

    Strength is deliberately moderate (1.5 / -0.5) to avoid introducing
    ringing artefacts that degrade ArcFace embedding accuracy.
    """
    import cv2

    blurred = cv2.GaussianBlur(img, (0, 0), 2.5)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

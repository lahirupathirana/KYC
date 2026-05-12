"""
Light image preparation for face recognition.

InsightFace (buffalo_l) was trained on naturalistic RGB images.
Heavy filters (CLAHE, sharpening, denoising) hurt embedding quality.
Only correct severe under/over-exposure before passing to the model.
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

"""
Document image augmentation pipeline.

Simulates real-world degradations found in mobile-captured ID photos:
  - Motion blur (hand shake)
  - Perspective warp (camera angle)
  - JPEG compression artefacts
  - Brightness/contrast variation (lighting)
  - Gaussian noise (camera sensor noise)
  - Shadow overlay (partial shadow)
  - Glare/highlight patch (glossy surface)

Usage:
    python scripts/dataset/augment.py \\
        --input datasets/synthetic/nic \\
        --output datasets/augmented/nic \\
        --copies 5 \\
        --seed 42

Requires:
    pip install opencv-python-headless numpy albumentations Pillow
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import albumentations as A
    _ALBUMENTATIONS = True
except ImportError:
    _ALBUMENTATIONS = False


# ── Augmentation pipeline ─────────────────────────────────────────────────────

def build_albumentations_pipeline() -> "A.Compose":
    """Heavy augmentation for training data diversity."""
    if not _ALBUMENTATIONS:
        raise ImportError("pip install albumentations")

    return A.Compose([
        # Geometric
        A.Perspective(scale=(0.02, 0.08), p=0.6),
        A.Rotate(limit=8, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03, scale_limit=0.05, rotate_limit=5,
            border_mode=cv2.BORDER_REPLICATE, p=0.4,
        ),

        # Blur / noise
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.5),
        A.GaussNoise(var_limit=(5, 40), mean=0, p=0.4),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.3), p=0.3),

        # Photometric
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3
        ),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05, p=0.3),

        # Compression
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.5),

        # Downscale (simulate low-res capture then upsample)
        A.Downscale(scale_min=0.5, scale_max=0.9, p=0.3),
    ])


def augment_opencv(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Fallback augmentation using only OpenCV (no Albumentations dependency).

    Lighter than the Albumentations pipeline but still useful.
    """
    # Motion blur
    if rng.random() < 0.4:
        size = rng.choice([3, 5, 7])
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1.0 / size
        if rng.random() < 0.5:
            kernel = kernel.T
        img = cv2.filter2D(img, -1, kernel)

    # Gaussian noise
    if rng.random() < 0.4:
        noise = np.random.normal(0, rng.uniform(3, 15), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Brightness / contrast
    if rng.random() < 0.6:
        alpha = rng.uniform(0.75, 1.35)  # contrast
        beta = rng.randint(-30, 30)       # brightness
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Perspective warp
    if rng.random() < 0.5:
        img = _random_perspective(img, rng, max_shift=0.07)

    # JPEG compression
    if rng.random() < 0.5:
        quality = rng.randint(50, 92)
        _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # Shadow overlay
    if rng.random() < 0.3:
        img = _add_shadow(img, rng)

    # Glare patch
    if rng.random() < 0.2:
        img = _add_glare(img, rng)

    return img


def _random_perspective(
    img: np.ndarray, rng: random.Random, max_shift: float
) -> np.ndarray:
    h, w = img.shape[:2]
    shift = max_shift

    def jitter(v: float) -> float:
        return v + rng.uniform(-shift, shift)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [jitter(0) * w, jitter(0) * h],
        [jitter(1) * w, jitter(0) * h],
        [jitter(1) * w, jitter(1) * h],
        [jitter(0) * w, jitter(1) * h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _add_shadow(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)

    x1 = rng.randint(0, w // 2)
    x2 = rng.randint(w // 2, w)
    shadow_intensity = rng.uniform(0.4, 0.75)

    pts = np.array([
        [x1, 0], [x2, 0], [w, h], [0, h]
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], shadow_intensity)

    return np.clip(img.astype(np.float32) * mask[:, :, np.newaxis], 0, 255).astype(np.uint8)


def _add_glare(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)

    cx = rng.randint(w // 4, 3 * w // 4)
    cy = rng.randint(h // 4, 3 * h // 4)
    radius = rng.randint(30, 120)
    intensity = rng.uniform(0.5, 1.0)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    glare_mask = np.clip(1.0 - dist / radius, 0, 1) * intensity

    out += glare_mask[:, :, np.newaxis] * 255
    return np.clip(out, 0, 255).astype(np.uint8)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Augment synthetic document images")
    parser.add_argument("--input", required=True, help="Input image directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--copies", type=int, default=5, help="Augmented copies per image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    use_albumentations = _ALBUMENTATIONS
    transform = build_albumentations_pipeline() if use_albumentations else None

    image_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    print(f"Found {len(image_files)} source images")
    print(f"Using {'Albumentations' if use_albumentations else 'OpenCV fallback'} pipeline")

    generated = 0
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  SKIP (unreadable): {img_path.name}", file=sys.stderr)
            continue

        # Copy annotations if present
        ann_path = img_path.with_suffix(".json")
        ann_data = json.loads(ann_path.read_text()) if ann_path.exists() else None

        for i in range(args.copies):
            if use_albumentations and transform is not None:
                aug_img = transform(image=img)["image"]
            else:
                aug_img = augment_opencv(img, rng)

            stem = img_path.stem
            out_name = f"{stem}_aug{i:03d}.jpg"
            cv2.imwrite(str(output_dir / out_name), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            if ann_data is not None:
                aug_ann = {**ann_data, "augmented": True, "source_image": img_path.name}
                (output_dir / out_name.replace(".jpg", ".json")).write_text(
                    json.dumps(aug_ann, indent=2)
                )

            generated += 1

    print(f"Generated {generated} augmented images → {output_dir}")


if __name__ == "__main__":
    main()

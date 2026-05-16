"""
Convert annotated synthetic images into PaddleOCR training format.

PaddleOCR expects:
  Detection (DB/DRRG):
    images/         — image files
    train_det.txt   — one line per image:
                       path/to/img.jpg\t[{"transcription":"txt","points":[[x,y],...]}]

  Recognition (CRNN/SVTR):
    rec_images/     — word-level image crops
    train_rec.txt   — one line per crop:
                       rec_images/crop_0001.jpg\tTEXT_CONTENT

This script reads the manifest.json files produced by generate_*.py and
produces the folder structure above, ready for PaddleOCR fine-tuning.

Usage:
    python scripts/train/prepare_paddleocr_dataset.py \\
        --input datasets/synthetic \\
        --output datasets/paddleocr \\
        --split 0.9

Requires:
    pip install Pillow numpy
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False


# ── Field keys to export as recognition crops ─────────────────────────────────
_FIELD_KEYS = ["document_number", "full_name", "dob", "expiry_date", "issue_date"]


def build_det_label(ann: dict, img_path: Path) -> str:
    """
    Build one detection label line: img_path\t[{...}]

    Converts each bounding box in the annotation into a 4-point polygon
    (PaddleOCR detection format).
    """
    gt = ann.get("ground_truth", {})
    bboxes = ann.get("bboxes", {})

    regions = []
    for key, bbox in bboxes.items():
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        text = gt.get(key, "")
        if not text:
            # Try fuzzy key match
            for fk in _FIELD_KEYS:
                if fk in key or key in fk:
                    text = gt.get(fk, "")
                    break
        if not text:
            continue

        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        regions.append({"transcription": str(text), "points": points})

    if not regions:
        return ""

    return f"{img_path.name}\t{json.dumps(regions, ensure_ascii=False)}"


def crop_recognition_images(
    img_path: Path, ann: dict, out_dir: Path
) -> list[tuple[str, str]]:
    """
    Crop word-level regions and return (crop_path, text) pairs.
    Only runs when PIL is available.
    """
    if not _PIL:
        return []

    img = Image.open(img_path)
    gt = ann.get("ground_truth", {})
    bboxes = ann.get("bboxes", {})
    pairs: list[tuple[str, str]] = []

    for key, bbox in bboxes.items():
        if len(bbox) != 4:
            continue
        text = gt.get(key, "")
        if not text:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img.crop((x1, y1, x2, y2))
        crop_name = f"{img_path.stem}_{key}.jpg"
        crop_path = out_dir / crop_name
        crop.save(str(crop_path), "JPEG", quality=90)
        pairs.append((f"rec_images/{crop_name}", str(text)))

    return pairs


def process_split(
    manifests: list[Path],
    split_ratio: float,
    out_dir: Path,
    seed: int,
) -> None:
    rng = random.Random(seed)

    det_images_dir = out_dir / "images"
    rec_images_dir = out_dir / "rec_images"
    det_images_dir.mkdir(parents=True, exist_ok=True)
    rec_images_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[tuple[Path, dict]] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        src_dir = manifest_path.parent
        for ann in manifest:
            doc_type = ann.get("document_type", "unknown")
            img_id = ann.get("id", 0)
            img_name = f"{doc_type}_{img_id:05d}.jpg"
            img_path = src_dir / img_name
            if img_path.exists():
                all_samples.append((img_path, ann))

    rng.shuffle(all_samples)
    n_train = int(len(all_samples) * split_ratio)
    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:],
    }

    for split_name, samples in splits.items():
        det_lines: list[str] = []
        rec_lines: list[str] = []

        for img_path, ann in samples:
            # Copy image to det_images/
            dest = det_images_dir / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)

            det_line = build_det_label(ann, dest)
            if det_line:
                det_lines.append(det_line)

            rec_pairs = crop_recognition_images(img_path, ann, rec_images_dir)
            rec_lines.extend(f"{p}\t{t}" for p, t in rec_pairs)

        (out_dir / f"{split_name}_det.txt").write_text(
            "\n".join(det_lines), encoding="utf-8"
        )
        (out_dir / f"{split_name}_rec.txt").write_text(
            "\n".join(rec_lines), encoding="utf-8"
        )

        print(f"  {split_name}: {len(det_lines)} det / {len(rec_lines)} rec samples")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare PaddleOCR detection+recognition dataset from synthetic manifests"
    )
    parser.add_argument("--input", default="datasets/synthetic",
                        help="Root directory containing nic/, passport/, driving_license/ sub-dirs")
    parser.add_argument("--output", default="datasets/paddleocr")
    parser.add_argument("--split", type=float, default=0.9, help="Train fraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    manifests = sorted(input_dir.rglob("manifest.json"))

    if not manifests:
        print(f"No manifest.json files found under {input_dir}")
        print("Run generate_nic.py / generate_passport.py / generate_driving_license.py first.")
        return

    print(f"Found {len(manifests)} manifests:")
    for m in manifests:
        print(f"  {m}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    process_split(manifests, args.split, out_dir, args.seed)
    print(f"\nDataset written to: {out_dir}")
    print("Next: run scripts/train/finetune_paddleocr.py to generate training configs")


if __name__ == "__main__":
    main()

"""
Synthetic Sri Lankan NIC image generator.

Generates fake NIC card images (both old and new format layouts) using PIL.
All data is fabricated — no real personal information is used.

Output per image:
  - <id>.jpg     — rendered card image
  - <id>.json    — annotation with ground-truth field values and bounding boxes

Usage:
    python scripts/dataset/generate_nic.py \\
        --output datasets/synthetic/nic \\
        --count 500 \\
        --seed 42

Requires:
    pip install Pillow numpy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow import of fake_data from scripts/dataset/ regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.dataset.fake_data import FakeSriLankan

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False


# ── Card layout constants (pixels, 300 DPI equivalent) ───────────────────────

CARD_W, CARD_H = 856, 540       # Standard CR80 card at 100 DPI
BG_COLOR = (240, 235, 220)      # Off-white card body
HEADER_COLOR = (20, 80, 160)    # Dark blue header bar
HEADER_H = 70
TEXT_COLOR = (20, 20, 20)
LABEL_COLOR = (80, 80, 80)
FONT_SIZE_TITLE = 18
FONT_SIZE_FIELD = 14
FONT_SIZE_LABEL = 11
FONT_SIZE_NIC = 20
PADDING = 40
LABEL_X = PADDING
VALUE_X = 200


def _load_font(size: int):
    """Load a monospace or default PIL font."""
    try:
        # Prefer a system monospace font for OCR training realism
        for name in [
            "DejaVuSansMono.ttf", "cour.ttf", "Courier New.ttf",
            "LiberationMono-Regular.ttf",
        ]:
            try:
                return ImageFont.truetype(name, size)
            except (IOError, OSError):
                continue
    except Exception:
        pass
    return ImageFont.load_default()


def render_new_nic(gen: FakeSriLankan, img_id: int) -> tuple[Image.Image, dict]:
    """Render a new-format (12-digit) NIC card."""
    dob = gen.dob()
    sex = gen.sex()
    name = gen.full_name()
    nic = gen.nic_number(dob=dob, sex=sex)
    address = gen.address()

    img = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([(0, 0), (CARD_W, HEADER_H)], fill=HEADER_COLOR)

    title_font = _load_font(FONT_SIZE_TITLE)
    label_font = _load_font(FONT_SIZE_LABEL)
    value_font = _load_font(FONT_SIZE_FIELD)
    nic_font = _load_font(FONT_SIZE_NIC)

    draw.text((PADDING, 15), "SRI LANKA", font=title_font, fill=(255, 255, 255))
    draw.text((PADDING, 38), "NATIONAL IDENTITY CARD", font=label_font, fill=(200, 220, 255))

    # Fields
    fields_data = [
        ("NIC Number", nic),
        ("Full Name", name),
        ("Date of Birth", dob.strftime("%Y-%m-%d")),
        ("Sex", "MALE" if sex == "M" else "FEMALE"),
        ("Address", address[:60]),  # truncate for layout
    ]

    bboxes: dict[str, list[int]] = {}
    y = HEADER_H + 20

    for label, value in fields_data:
        draw.text((LABEL_X, y), f"{label}:", font=label_font, fill=LABEL_COLOR)
        font = nic_font if label == "NIC Number" else value_font
        draw.text((VALUE_X, y - 2), value, font=font, fill=TEXT_COLOR)

        # Approximate bounding box for value text (for annotation)
        bbox = [VALUE_X, y - 2, VALUE_X + len(value) * 9, y + FONT_SIZE_FIELD + 4]
        bboxes[label.lower().replace(" ", "_")] = bbox

        y += 52

    # Decorative elements (stripe)
    draw.rectangle([(0, CARD_H - 40), (CARD_W, CARD_H)], fill=HEADER_COLOR)
    draw.text((PADDING, CARD_H - 28), "DEMOCRATIC SOCIALIST REPUBLIC OF SRI LANKA",
              font=label_font, fill=(200, 220, 255))

    annotation = {
        "id": img_id,
        "document_type": "nic",
        "format": "new",
        "ground_truth": {
            "document_number": nic,
            "full_name": name,
            "dob": dob.strftime("%Y-%m-%d"),
            "sex": sex,
            "address": address,
        },
        "bboxes": bboxes,
        "image_size": [CARD_W, CARD_H],
    }

    return img, annotation


def render_old_nic(gen: FakeSriLankan, img_id: int) -> tuple[Image.Image, dict]:
    """Render an old-format (9-digit+V/X) NIC card."""
    dob = gen.dob(min_age=35, max_age=70)
    sex = gen.sex()
    name = gen.full_name()
    nic = gen.old_nic_number(dob=dob, sex=sex)

    img = Image.new("RGB", (CARD_W, CARD_H), (230, 225, 210))
    draw = ImageDraw.Draw(img)

    draw.rectangle([(0, 0), (CARD_W, HEADER_H)], fill=(10, 60, 130))

    title_font = _load_font(FONT_SIZE_TITLE)
    label_font = _load_font(FONT_SIZE_LABEL)
    value_font = _load_font(FONT_SIZE_FIELD)
    nic_font = _load_font(FONT_SIZE_NIC)

    draw.text((PADDING, 15), "SRI LANKA", font=title_font, fill=(255, 255, 255))
    draw.text((PADDING, 38), "IDENTITY CARD  / හැඳුනුම්පත", font=label_font, fill=(200, 220, 255))

    fields_data = [
        ("NIC No", nic),
        ("Name / නම", name),
        ("D.O.B.", dob.strftime("%Y-%m-%d")),
        ("Sex / ස්ත්‍රී පු.", "MALE / පුරුෂ" if sex == "M" else "FEMALE / ස්ත්‍රී"),
    ]

    bboxes: dict[str, list[int]] = {}
    y = HEADER_H + 20

    for label, value in fields_data:
        draw.text((LABEL_X, y), f"{label}:", font=label_font, fill=LABEL_COLOR)
        font = nic_font if "NIC" in label else value_font
        draw.text((VALUE_X, y - 2), value, font=font, fill=TEXT_COLOR)
        key = label.split("/")[0].strip().lower().replace(". ", "_").replace(" ", "_")
        bboxes[key] = [VALUE_X, y - 2, VALUE_X + len(value) * 9, y + FONT_SIZE_FIELD + 4]
        y += 52

    annotation = {
        "id": img_id,
        "document_type": "nic",
        "format": "old",
        "ground_truth": {
            "document_number": nic,
            "full_name": name,
            "dob": dob.strftime("%Y-%m-%d"),
            "sex": sex,
        },
        "bboxes": bboxes,
        "image_size": [CARD_W, CARD_H],
    }

    return img, annotation


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _PIL:
        print("ERROR: Pillow is required. Run: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate synthetic NIC images")
    parser.add_argument("--output", default="datasets/synthetic/nic", help="Output directory")
    parser.add_argument("--count", type=int, default=200, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = FakeSriLankan(seed=args.seed)
    annotations = []

    for i in range(args.count):
        # Mix old and new NIC formats
        if i % 4 == 0:
            img, ann = render_old_nic(gen, i)
        else:
            img, ann = render_new_nic(gen, i)

        img_path = out_dir / f"nic_{i:05d}.jpg"
        ann_path = out_dir / f"nic_{i:05d}.json"

        img.save(str(img_path), "JPEG", quality=92)
        ann_path.write_text(json.dumps(ann, indent=2, ensure_ascii=False))
        annotations.append(ann)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{args.count}")

    # Write combined manifest
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(annotations, indent=2, ensure_ascii=False))
    print(f"Done. {args.count} NIC images → {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

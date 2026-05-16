"""
Synthetic Sri Lankan Driving License image generator.

Generates DL card images with English text layout.
All data is fabricated.

Usage:
    python scripts/dataset/generate_driving_license.py \\
        --output datasets/synthetic/driving_license \\
        --count 200 \\
        --seed 42

Requires:
    pip install Pillow numpy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.dataset.fake_data import FakeSriLankan

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False


# ── Layout constants ──────────────────────────────────────────────────────────

CARD_W, CARD_H = 856, 540
BG_COLOR = (245, 242, 230)
HEADER_COLOR = (180, 0, 0)   # Red — SL DL uses red header
TEXT_COLOR = (10, 10, 10)
LABEL_COLOR = (80, 80, 80)
HEADER_H = 65
PADDING = 35
VALUE_X = 210
FONT_SIZE_TITLE = 17
FONT_SIZE_LABEL = 11
FONT_SIZE_VALUE = 14
FONT_SIZE_NUMBER = 20


def _load_font(size: int):
    for name in [
        "DejaVuSans.ttf", "arial.ttf", "Arial.ttf",
        "LiberationSans-Regular.ttf", "DejaVuSansMono.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def render_driving_license(gen: FakeSriLankan, img_id: int) -> tuple[Image.Image, dict]:
    dob = gen.dob(min_age=18, max_age=65)
    sex = gen.sex()
    name = gen.full_name()
    dl_no = gen.driving_license_number()
    issue = gen.issue_date(dob=dob)
    expiry = gen.dl_expiry(issued=issue)
    address = gen.address()
    categories = gen.vehicle_categories()

    img = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(0, 0), (CARD_W, HEADER_H)], fill=HEADER_COLOR)
    title_font = _load_font(FONT_SIZE_TITLE)
    label_font = _load_font(FONT_SIZE_LABEL)
    value_font = _load_font(FONT_SIZE_VALUE)
    no_font = _load_font(FONT_SIZE_NUMBER)

    draw.text((PADDING, 12), "DEMOCRATIC SOCIALIST REPUBLIC OF SRI LANKA",
              font=label_font, fill=(255, 220, 220))
    draw.text((PADDING, 35), "DEPARTMENT OF MOTOR TRAFFIC — DRIVING LICENCE",
              font=title_font, fill=(255, 255, 255))

    # Fields
    fields = [
        ("Licence No.", dl_no),
        ("Licence Holder", name),
        ("Date of Birth", dob.strftime("%d/%m/%Y")),
        ("Date of Issue", issue.strftime("%d/%m/%Y")),
        ("Expiry Date", expiry.strftime("%d/%m/%Y")),
        ("Address", address[:58]),
        ("Vehicle Category", " / ".join(categories)),
    ]

    bboxes: dict[str, list[int]] = {}
    y = HEADER_H + 15

    for label, value in fields:
        draw.text((PADDING, y), f"{label}:", font=label_font, fill=LABEL_COLOR)
        font = no_font if "Licence No" in label else value_font
        draw.text((VALUE_X, y - 2), value, font=font, fill=TEXT_COLOR)
        key = label.lower().replace(" ", "_").replace(".", "")
        bboxes[key] = [VALUE_X, y - 2, VALUE_X + len(value) * 9, y + 18]
        y += 50

    # Footer bar
    draw.rectangle([(0, CARD_H - 35), (CARD_W, CARD_H)], fill=HEADER_COLOR)
    draw.text((PADDING, CARD_H - 24), "MOTOR TRAFFIC ACT NO. 14 OF 1951",
              font=label_font, fill=(255, 200, 200))

    annotation = {
        "id": img_id,
        "document_type": "driving_license",
        "ground_truth": {
            "document_number": dl_no,
            "full_name": name,
            "dob": dob.strftime("%Y-%m-%d"),
            "sex": sex,
            "issue_date": issue.strftime("%Y-%m-%d"),
            "expiry_date": expiry.strftime("%Y-%m-%d"),
            "address": address,
            "vehicle_categories": categories,
        },
        "bboxes": bboxes,
        "image_size": [CARD_W, CARD_H],
    }

    return img, annotation


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _PIL:
        print("ERROR: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate synthetic driving license images")
    parser.add_argument("--output", default="datasets/synthetic/driving_license")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = FakeSriLankan(seed=args.seed)
    annotations = []

    for i in range(args.count):
        img, ann = render_driving_license(gen, i)
        img_path = out_dir / f"dl_{i:05d}.jpg"
        ann_path = out_dir / f"dl_{i:05d}.json"
        img.save(str(img_path), "JPEG", quality=92)
        ann_path.write_text(json.dumps(ann, indent=2, ensure_ascii=False))
        annotations.append(ann)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{args.count}")

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(annotations, indent=2, ensure_ascii=False))
    print(f"Done. {args.count} DL images → {out_dir}")


if __name__ == "__main__":
    main()

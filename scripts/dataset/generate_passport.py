"""
Synthetic Sri Lankan passport page generator.

Generates the biographic data page of a Sri Lankan passport, including a
simulated MRZ zone. All data is fabricated.

Output per image:
  - <id>.jpg  — rendered passport page
  - <id>.json — ground-truth annotation with MRZ lines and field values

Usage:
    python scripts/dataset/generate_passport.py \\
        --output datasets/synthetic/passport \\
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

PAGE_W, PAGE_H = 1240, 877   # A5 landscape at 150 DPI (approx passport page)
BG_COLOR = (248, 246, 240)   # Off-white passport page
HEADER_COLOR = (10, 56, 120)
MRZ_BG = (230, 230, 225)
TEXT_COLOR = (10, 10, 10)
LABEL_COLOR = (90, 90, 90)
MRZ_COLOR = (15, 15, 15)
PADDING = 60
LABEL_W = 220


def _load_font(size: int, mono: bool = False):
    mono_candidates = [
        "DejaVuSansMono.ttf", "cour.ttf", "Courier New.ttf",
        "LiberationMono-Regular.ttf", "Consolas.ttf",
    ]
    prop_candidates = [
        "DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf",
    ]
    candidates = mono_candidates if mono else prop_candidates
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def render_passport(gen: FakeSriLankan, img_id: int) -> tuple[Image.Image, dict]:
    dob = gen.dob(min_age=18, max_age=65)
    sex = gen.sex()
    name = gen.full_name()
    surname = name.split()[-1].upper() if " " in name else name.upper()
    given = " ".join(name.split()[:-1]).upper() if " " in name else ""
    passport_no = gen.passport_number()
    expiry = gen.passport_expiry()
    nationality = "LKA"
    country = "LKA"

    mrz1 = gen.mrz_line1(surname, given, country)
    mrz2 = gen.mrz_line2(passport_no, nationality, dob, sex, expiry)

    img = Image.new("RGB", (PAGE_W, PAGE_H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Header band
    draw.rectangle([(0, 0), (PAGE_W, 80)], fill=HEADER_COLOR)

    header_font = _load_font(22)
    sub_font = _load_font(13)
    label_font = _load_font(12)
    value_font = _load_font(16)
    mrz_font = _load_font(18, mono=True)

    draw.text((PADDING, 18), "DEMOCRATIC SOCIALIST REPUBLIC OF SRI LANKA",
              font=header_font, fill=(255, 255, 255))
    draw.text((PADDING, 48), "PASSPORT / ගමන් බලපත්‍රය / கடவுச்சீட்டு",
              font=sub_font, fill=(200, 220, 255))

    # Biographic fields
    fields = [
        ("Surname / නාමය", surname),
        ("Given Names / ලාබ නාමය", given or "—"),
        ("Passport No.", passport_no),
        ("Nationality", nationality),
        ("Date of Birth", dob.strftime("%d %b %Y").upper()),
        ("Sex / ස්ත්‍රී පු.", "M" if sex == "M" else "F"),
        ("Date of Expiry", expiry.strftime("%d %b %Y").upper()),
    ]

    bboxes: dict[str, list[int]] = {}
    y = 100

    for label, value in fields:
        draw.text((PADDING, y), label, font=label_font, fill=LABEL_COLOR)
        draw.text((PADDING + LABEL_W, y), value, font=value_font, fill=TEXT_COLOR)
        bboxes[label.split("/")[0].strip().lower().replace(" ", "_").replace(".", "")] = [
            PADDING + LABEL_W, y, PADDING + LABEL_W + len(value) * 10, y + 20
        ]
        y += 70

    # MRZ zone
    mrz_y = PAGE_H - 120
    draw.rectangle([(0, mrz_y - 10), (PAGE_W, PAGE_H)], fill=MRZ_BG)
    draw.text((PADDING, mrz_y), mrz1, font=mrz_font, fill=MRZ_COLOR)
    draw.text((PADDING, mrz_y + 40), mrz2, font=mrz_font, fill=MRZ_COLOR)

    bboxes["mrz_line1"] = [PADDING, mrz_y, PADDING + len(mrz1) * 11, mrz_y + 30]
    bboxes["mrz_line2"] = [PADDING, mrz_y + 40, PADDING + len(mrz2) * 11, mrz_y + 70]

    annotation = {
        "id": img_id,
        "document_type": "passport",
        "ground_truth": {
            "document_number": passport_no,
            "full_name": name,
            "surname": surname,
            "given_names": given,
            "dob": dob.strftime("%Y-%m-%d"),
            "sex": sex,
            "nationality": nationality,
            "expiry_date": expiry.strftime("%Y-%m-%d"),
            "mrz_line1": mrz1,
            "mrz_line2": mrz2,
        },
        "bboxes": bboxes,
        "image_size": [PAGE_W, PAGE_H],
    }

    return img, annotation


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _PIL:
        print("ERROR: Pillow required. Run: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate synthetic passport page images")
    parser.add_argument("--output", default="datasets/synthetic/passport")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = FakeSriLankan(seed=args.seed)
    annotations = []

    for i in range(args.count):
        img, ann = render_passport(gen, i)
        img_path = out_dir / f"passport_{i:05d}.jpg"
        ann_path = out_dir / f"passport_{i:05d}.json"
        img.save(str(img_path), "JPEG", quality=92)
        ann_path.write_text(json.dumps(ann, indent=2, ensure_ascii=False))
        annotations.append(ann)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{args.count}")

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(annotations, indent=2, ensure_ascii=False))
    print(f"Done. {args.count} passport images → {out_dir}")


if __name__ == "__main__":
    main()

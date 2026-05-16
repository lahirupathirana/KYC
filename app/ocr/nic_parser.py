"""
Sri Lankan National Identity Card (NIC) field extractor.

Old format  9 digits + V or X  e.g. 890123456V
  Digits 0-1  : last two digits of birth year  (19XX)
  Digits 2-4  : day-of-year  (add 500 for female)
  Digits 5-8  : serial number

New format  12 digits            e.g. 198901234567
  Digits 0-3  : full birth year
  Digits 4-6  : day-of-year  (add 500 for female)
  Digits 7-11 : serial + check digit

DOB is always derivable from the NIC number itself when the format is recognised.
Explicit DOB fields on the card are extracted as a secondary source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from datetime import datetime


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class NICFields:
    document_number: str | None = None
    full_name: str | None = None
    dob: str | None = None    # YYYY-MM-DD
    sex: str | None = None    # M / F
    confidence: float = 0.0


# ── Patterns ──────────────────────────────────────────────────────────────────

# Use (?<!\d) / (?!\d) instead of \b so digits embedded in text like
# "No198726501608" are still matched (o→1 has no \b word boundary).
_OLD_NIC_RE = re.compile(r"(?<!\d)(\d{9})[VvXx](?!\w)")
_NEW_NIC_RE = re.compile(r"(?<!\d)(\d{12})(?!\d)")

_DATE_LABEL_RE = re.compile(
    r"(?:DATE\s*OF\s*BIRTH|D\.O\.B\.?|DOB|BORN)[:\s]*"
    r"(\d{1,2}[./ -]\d{1,2}[./ -]\d{2,4}|\d{8})",
    re.IGNORECASE,
)

# Sex keywords — covers OCR garbling of "Male"/"Female"
_MALE_RE = re.compile(r"\bM(?:ale?|als?|aie?)\b", re.IGNORECASE)
_FEMALE_RE = re.compile(r"\bF(?:emale?|emais?|emai[ls]?)\b", re.IGNORECASE)

_NAME_LABELS = {
    "NAME", "FULL NAME", "NAME IN FULL", "SURNAME",
    "GIVEN NAMES", "INITIALS AND SURNAME",
}

# OCR often glues the label to the value with a colon — split on it
_LABEL_COLON_RE = re.compile(r"^(?:Name|Full\s*Name|Initials.*Surname)\s*:\s*", re.IGNORECASE)

# Keywords that should not be treated as names
_NON_NAME_WORDS = {
    "REPUBLIC", "IDENTITY", "NATIONAL", "CARD", "SRI", "LANKA",
    "DEMOCRATIC", "SOCIALIST", "SIGNATURE", "HOLDER",
}


# ── Public entry point ────────────────────────────────────────────────────────

def parse(text_blocks: list[dict]) -> NICFields | None:
    texts = [b["text"] for b in text_blocks]
    joined = " ".join(texts)

    nic_number, dob_from_nic, sex_from_nic = _extract_nic_number(joined)
    name = _extract_name(texts)
    sex = sex_from_nic or _extract_sex(joined)
    dob = dob_from_nic or _extract_explicit_dob(joined)

    if not any([nic_number, name, dob]):
        return None

    filled = sum(1 for v in [nic_number, name, dob] if v)
    return NICFields(
        document_number=nic_number,
        full_name=name,
        dob=dob,
        sex=sex,
        confidence=round(filled / 3, 2),
    )


# ── NIC number extraction and decoding ───────────────────────────────────────

def _extract_nic_number(text: str) -> tuple[str | None, str | None, str | None]:
    # Old format takes priority (more distinctive due to V/X suffix)
    m = _OLD_NIC_RE.search(text)
    if m:
        digits = m.group(1)
        nic_no = m.group(0).upper()
        dob, sex = _decode_old_nic(digits)
        return nic_no, dob, sex

    # New 12-digit format — scan all matches and pick the first valid one
    for m in _NEW_NIC_RE.finditer(text):
        digits = m.group(1)
        year_candidate = int(digits[:4])
        day_candidate = int(digits[4:7])
        effective_day = day_candidate - 500 if day_candidate > 500 else day_candidate
        if (1900 <= year_candidate <= 2025) and (1 <= effective_day <= 366):
            dob, sex = _decode_new_nic(digits)
            return digits, dob, sex

    return None, None, None


def _decode_old_nic(digits: str) -> tuple[str | None, str | None]:
    try:
        year = 1900 + int(digits[:2])
        raw_day = int(digits[2:5])
        sex = "F" if raw_day > 500 else "M"
        day_of_year = raw_day - 500 if sex == "F" else raw_day
        if not (1 <= day_of_year <= 366):
            return None, sex
        birth = date(year, 1, 1) + timedelta(days=day_of_year - 1)
        return birth.strftime("%Y-%m-%d"), sex
    except (ValueError, OverflowError):
        return None, None


def _decode_new_nic(digits: str) -> tuple[str | None, str | None]:
    try:
        year = int(digits[:4])
        raw_day = int(digits[4:7])
        sex = "F" if raw_day > 500 else "M"
        day_of_year = raw_day - 500 if sex == "F" else raw_day
        if not (1 <= day_of_year <= 366):
            return None, sex
        birth = date(year, 1, 1) + timedelta(days=day_of_year - 1)
        return birth.strftime("%Y-%m-%d"), sex
    except (ValueError, OverflowError):
        return None, None


# ── Name extraction ───────────────────────────────────────────────────────────

def _extract_name(texts: list[str]) -> str | None:
    # Strategy 1: label and value in the SAME block e.g. "Name: PATHIRANAGE LAHIRU SAMAN"
    for t in texts:
        after = _LABEL_COLON_RE.sub("", t).strip()
        if after != t.strip() and _is_plausible_name(after):
            return _clean_name(after)

    # Strategy 2: label in one block, value in the next
    for i, t in enumerate(texts):
        upper = t.upper().strip()
        if upper in _NAME_LABELS or any(lbl in upper for lbl in _NAME_LABELS):
            # Also accept inline value after colon even if label check fires
            inline = re.split(r"[:\-]\s*", t, maxsplit=1)
            if len(inline) > 1 and _is_plausible_name(inline[1]):
                return _clean_name(inline[1])
            if i + 1 < len(texts) and _is_plausible_name(texts[i + 1]):
                return _clean_name(texts[i + 1])

    # Strategy 3: longest all-alpha block that looks like a proper name
    candidates = [
        t.strip() for t in texts
        if re.match(r"^[A-Z][A-Za-z .]{4,60}$", t.strip())
        and not any(kw in t.upper().split() for kw in _NON_NAME_WORDS)
    ]
    return _clean_name(max(candidates, key=len)) if candidates else None


def _clean_name(raw: str) -> str:
    # Remove any leading noise characters (digits, punctuation)
    cleaned = re.sub(r"^[^A-Za-z]+", "", raw).strip()
    # Collapse multiple spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.title() if cleaned else raw.title()


def _is_plausible_name(text: str) -> bool:
    t = text.strip()
    return bool(re.match(r"^[A-Za-z][A-Za-z\s.\-]{2,60}$", t)) and len(t.split()) >= 1


# ── Sex extraction ────────────────────────────────────────────────────────────

def _extract_sex(text: str) -> str | None:
    """
    Find M/F from OCR text.

    Handles garbled OCR: "Mals", "Mais", "Femais", "/SexMals", etc.
    Sex derived from the NIC number itself (in _extract_nic_number) takes
    priority over this text-based extraction.
    """
    if _FEMALE_RE.search(text):
        return "F"
    if _MALE_RE.search(text):
        return "M"
    # Fallback: explicit single-letter field label
    if re.search(r"\bSex\s*[:\-]?\s*F\b", text, re.IGNORECASE):
        return "F"
    if re.search(r"\bSex\s*[:\-]?\s*M\b", text, re.IGNORECASE):
        return "M"
    return None


# ── Explicit DOB extraction (fallback) ───────────────────────────────────────

def _extract_explicit_dob(text: str) -> str | None:
    m = _DATE_LABEL_RE.search(text)
    if not m:
        return None
    return _normalise_date(m.group(1).strip())


def _normalise_date(raw: str) -> str | None:
    cleaned = re.sub(r"[\s]", "/", raw.strip())
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

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

_OLD_NIC_RE = re.compile(r"\b(\d{9})[VvXx]\b")
_NEW_NIC_RE = re.compile(r"\b(\d{12})\b")

_DATE_LABEL_RE = re.compile(
    r"(?:DATE\s+OF\s+BIRTH|D\.O\.B\.?|DOB|BORN)[:\s]*"
    r"(\d{1,2}[./ -]\d{1,2}[./ -]\d{2,4}|\d{8})",
    re.IGNORECASE,
)

_NAME_LABELS = {"NAME", "FULL NAME", "NAME IN FULL", "SURNAME", "GIVEN NAMES", "INITIALS AND SURNAME"}


# ── Public entry point ────────────────────────────────────────────────────────

def parse(text_blocks: list[dict]) -> NICFields | None:
    texts = [b["text"] for b in text_blocks]
    joined = " ".join(texts)

    nic_number, dob_from_nic, sex = _extract_nic_number(joined)
    name = _extract_name(texts)
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

    m = _NEW_NIC_RE.search(text)
    if m:
        digits = m.group(1)
        # Reject accidental 12-digit phone numbers or other sequences:
        # valid NIC new format has year 1900-2010 and day 001-866 (500+366)
        year_candidate = int(digits[:4])
        day_candidate = int(digits[4:7])
        if not (1900 <= year_candidate <= 2010 and 1 <= (day_candidate % 500 or day_candidate) <= 366):
            return None, None, None
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
    # Strategy 1: the token immediately after a NAME label
    for i, t in enumerate(texts):
        upper = t.upper().strip()
        if upper in _NAME_LABELS or any(lbl in upper for lbl in _NAME_LABELS):
            if i + 1 < len(texts):
                candidate = texts[i + 1].strip()
                if _is_plausible_name(candidate):
                    return candidate.title()

    # Strategy 2: longest all-alpha run that looks like a proper name
    candidates = [
        t.strip() for t in texts
        if re.match(r"^[A-Z][A-Za-z .]{4,50}$", t.strip())
        and not any(kw in t.upper() for kw in {"REPUBLIC", "IDENTITY", "NATIONAL", "CARD"})
    ]
    return max(candidates, key=len).title() if candidates else None


def _is_plausible_name(text: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z\s.\-]{2,50}$", text.strip()))


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

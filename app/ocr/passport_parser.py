"""
ICAO 9303 Machine Readable Passport (TD3) parser + heuristic fallback.

MRZ TD3 layout (passports — two 44-character lines):
  Line 1:  P<ISOCOUNTRYNAME<<FIRSTNAME<MIDDLE<<<<<<<<<<<<<<  (doc-type, country, names)
  Line 2:  DOCNUM9CNATSEX0DOBCHKEXPCHKPERSONALCHKCHK         (number, dates, check digits)

Character set: A–Z, 0–9, '<' (filler/separator).

Primary strategy: locate MRZ lines in OCR output → parse spec fields.
Fallback: regex over free text when MRZ is absent or unreadable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime


# ── Data model (internal — not Pydantic) ─────────────────────────────────────

@dataclass
class PassportFields:
    document_number: str | None = None
    surname: str | None = None
    given_names: str | None = None
    full_name: str | None = None
    dob: str | None = None          # YYYY-MM-DD
    sex: str | None = None          # M / F
    nationality: str | None = None
    expiry_date: str | None = None  # YYYY-MM-DD
    country_code: str | None = None
    mrz_parsed: bool = False
    confidence: float = 0.0


# ── Public entry point ────────────────────────────────────────────────────────

def parse(text_blocks: list[dict]) -> PassportFields | None:
    """
    Attempt MRZ extraction first; fall back to heuristic pattern matching.
    Returns None only if no passport-related data at all could be found.
    """
    texts = [b["text"] for b in text_blocks]
    full_text = "\n".join(texts)

    result = _parse_mrz(texts)
    if result and result.confidence >= 0.5:
        return result

    return _parse_heuristic(texts, full_text) or result  # return partial MRZ if heuristic also fails


# ── MRZ parsing ───────────────────────────────────────────────────────────────

_MRZ_CHAR_RE = re.compile(r"^[A-Z0-9<]{30,}$")

def _parse_mrz(texts: list[str]) -> PassportFields | None:
    # Normalise each block: uppercase, collapse spaces to <
    candidates = []
    for t in texts:
        cleaned = t.upper().strip().replace(" ", "<").replace("«", "<")
        # Pad/strip to exactly 44 if within 2 chars (OCR often drops trailing <)
        if 42 <= len(cleaned) <= 46 and _MRZ_CHAR_RE.match(cleaned):
            candidates.append(cleaned[:44].ljust(44, "<"))

    if len(candidates) < 2:
        return None

    # Try every consecutive pair (MRZ lines may not be adjacent in block order)
    for i in range(len(candidates) - 1):
        result = _try_parse_pair(candidates[i], candidates[i + 1])
        if result:
            return result

    return None


def _try_parse_pair(line1: str, line2: str) -> PassportFields | None:
    if line1[0] not in ("P", "V", "A", "C", "I"):
        return None
    try:
        country = line1[2:5].replace("<", "").strip()
        name_field = line1[5:44]
        parts = name_field.split("<<", 1)
        surname = parts[0].replace("<", " ").strip() if parts else None
        given_names = parts[1].replace("<", " ").strip() if len(parts) > 1 else None
        full_name = f"{given_names} {surname}".strip() if given_names and surname else (surname or given_names)

        doc_number = line2[0:9].replace("<", "").strip() or None
        nationality = line2[10:13].replace("<", "").strip() or None
        dob = _parse_mrz_date(line2[13:19], is_dob=True)
        sex_char = line2[19] if len(line2) > 19 else None
        sex = sex_char if sex_char in ("M", "F") else None
        expiry = _parse_mrz_date(line2[20:26], is_dob=False)

        filled = sum(1 for v in [doc_number, full_name, dob] if v)
        confidence = round(filled / 3, 2)

        return PassportFields(
            document_number=doc_number,
            surname=surname,
            given_names=given_names,
            full_name=full_name,
            dob=dob,
            sex=sex,
            nationality=nationality,
            expiry_date=expiry,
            country_code=country,
            mrz_parsed=True,
            confidence=confidence,
        )
    except (IndexError, ValueError):
        return None


def _parse_mrz_date(raw: str, *, is_dob: bool) -> str | None:
    if not re.match(r"^\d{6}$", raw):
        return None
    yy, mm, dd = int(raw[:2]), int(raw[2:4]), int(raw[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    current_yy = datetime.now().year % 100
    if is_dob:
        full_year = 2000 + yy if yy <= current_yy else 1900 + yy
    else:
        full_year = 2000 + yy if yy <= 50 else 1900 + yy
    return f"{full_year:04d}-{mm:02d}-{dd:02d}"


# ── Heuristic fallback ────────────────────────────────────────────────────────

_PASSPORT_NUM_RE = re.compile(r"\b([A-Z]{1,2}\d{6,8})\b")
_DATE_LABEL_RE = re.compile(
    r"(?:DATE\s+OF\s+BIRTH|D\.O\.B\.?|DOB|BIRTH\s+DATE)[:\s]*"
    r"(\d{1,2}[./ -]\d{1,2}[./ -]\d{2,4}|\d{8})",
    re.IGNORECASE,
)
_EXPIRY_LABEL_RE = re.compile(
    r"(?:EXPIRY|EXPIRATION|VALID\s+UNTIL|DATE\s+OF\s+EXPIRY)[:\s]*"
    r"(\d{1,2}[./ -]\d{1,2}[./ -]\d{2,4}|\d{8})",
    re.IGNORECASE,
)
_NAME_LABELS = {"SURNAME", "LAST NAME", "FAMILY NAME", "GIVEN NAMES", "FIRST NAME", "NAME"}


def _parse_heuristic(texts: list[str], full_text: str) -> PassportFields | None:
    upper = full_text.upper()

    doc_num_m = _PASSPORT_NUM_RE.search(upper)
    doc_number = doc_num_m.group(1) if doc_num_m else None

    dob = _extract_labelled_date(_DATE_LABEL_RE, full_text)
    expiry = _extract_labelled_date(_EXPIRY_LABEL_RE, full_text)
    name = _extract_labelled_name(texts)

    if not any([doc_number, dob, name]):
        return None

    filled = sum(1 for v in [doc_number, name, dob] if v)
    return PassportFields(
        document_number=doc_number,
        full_name=name,
        dob=dob,
        expiry_date=expiry,
        mrz_parsed=False,
        confidence=round(filled / 3 * 0.6, 2),  # lower confidence for heuristic
    )


def _extract_labelled_date(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text)
    if not m:
        return None
    return _normalise_date(m.group(1).strip())


def _normalise_date(raw: str) -> str | None:
    cleaned = re.sub(r"[\s]", "/", raw)
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _extract_labelled_name(texts: list[str]) -> str | None:
    for i, t in enumerate(texts):
        if t.upper().strip() in _NAME_LABELS or any(lbl in t.upper() for lbl in _NAME_LABELS):
            if i + 1 < len(texts):
                candidate = texts[i + 1].strip()
                if re.match(r"^[A-Za-z][\w\s\-]{2,50}$", candidate):
                    return candidate.title()
    return None

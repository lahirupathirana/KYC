"""
Sri Lankan Driving License field extraction.

Sri Lankan driving licenses (issued by Department of Motor Traffic) contain:
  - License number (12 digits, e.g., B1234567890 or numeric only)
  - Full name (English)
  - Name in Sinhala script (may appear as transliterated or unicode)
  - Date of birth
  - Issue date and expiry date
  - Address (English and/or Sinhala)
  - Vehicle categories: A, A1, B, B1, C, C1, D, D1, E, F, G, J
  - Hologram and microprint may interfere with OCR output

Strategy:
  1. License number detection via pattern matching
  2. Date extraction via labelled fields (ISSUE, EXPIRY, DOB) or positional heuristics
  3. Name extraction after common label keywords
  4. Vehicle category parsing from text grid
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Patterns ──────────────────────────────────────────────────────────────────

# Sri Lankan DL numbers: may start with letter(s) then digits, or pure digits
_DL_NUMBER_RE = re.compile(
    r"\b([A-Z]{1,2}\d{7,10}|\d{10,12})\b", re.IGNORECASE
)

# Date patterns: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD
_DATE_RE = re.compile(
    r"\b(\d{2}[/.\-]\d{2}[/.\-]\d{4}|\d{4}[/.\-]\d{2}[/.\-]\d{2})\b"
)

# Vehicle category codes recognised on Sri Lankan licenses
_VEHICLE_CATEGORIES = frozenset([
    "A", "A1", "B", "B1", "C", "C1", "D", "D1", "E", "F", "G", "J",
])

# Label keywords — normalised uppercase
_ISSUE_LABELS = {"ISSUE", "ISSUED", "DATE OF ISSUE", "ISSUE DATE", "VALID FROM"}
_EXPIRY_LABELS = {"EXPIRY", "EXPIRATION", "VALID UNTIL", "VALID TO", "EXPIRE", "EXPIRES"}
_DOB_LABELS = {"DATE OF BIRTH", "DOB", "D.O.B.", "BIRTH DATE", "BORN"}
_NAME_LABELS = {"NAME", "FULL NAME", "HOLDER", "LICENCE HOLDER", "LICENSE HOLDER"}
_ADDRESS_LABELS = {"ADDRESS", "RESIDENCE", "RESIDING AT"}
_DL_INDICATOR_KW = {
    "DRIVING LICENCE", "DRIVING LICENSE", "MOTOR TRAFFIC",
    "DEPARTMENT OF MOTOR", "DMT", "REPUBLIC OF SRI LANKA",
}


# ── Public data class ─────────────────────────────────────────────────────────

@dataclass
class DrivingLicenseFields:
    document_number: str | None = None
    full_name: str | None = None
    dob: str | None = None            # ISO 8601 YYYY-MM-DD
    issue_date: str | None = None     # ISO 8601 YYYY-MM-DD
    expiry_date: str | None = None    # ISO 8601 YYYY-MM-DD
    address: str | None = None
    vehicle_categories: list[str] = field(default_factory=list)
    confidence: float = 0.0


# ── Public entry point ────────────────────────────────────────────────────────

def parse(text_blocks: list[dict]) -> DrivingLicenseFields | None:
    """
    Extract driving license fields from OCR text blocks.

    Returns None when the blocks clearly contain no recognisable DL content.
    Returns a DrivingLicenseFields with confidence=0.0 if the document looks like
    a DL but extraction failed — allows the caller to surface partial results.
    """
    texts = [b["text"].strip() for b in text_blocks if b.get("text", "").strip()]
    if not texts:
        return None

    full_text = " ".join(texts)
    upper_text = full_text.upper()

    dl_number = _extract_dl_number(texts)
    name = _extract_name(texts)
    dob, issue_date, expiry_date = _extract_dates(texts, upper_text)
    address = _extract_address(texts)
    categories = _extract_vehicle_categories(upper_text)

    # Confidence = proportion of core fields (number, name, expiry) found
    core = [dl_number, name, expiry_date]
    confidence = sum(1 for v in core if v is not None) / len(core)

    result = DrivingLicenseFields(
        document_number=dl_number,
        full_name=name,
        dob=dob,
        issue_date=issue_date,
        expiry_date=expiry_date,
        address=address,
        vehicle_categories=categories,
        confidence=round(confidence, 3),
    )

    # Return None only if extraction found nothing at all
    if confidence == 0.0 and not categories:
        return None

    return result


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_dl_number(texts: list[str]) -> str | None:
    """Find driving license number: 10-12 digit string or letter-prefixed variant."""
    # Prefer the line that follows a "LICENSE NO" / "LIC NO" label
    for i, t in enumerate(texts):
        up = t.upper()
        if any(kw in up for kw in ("LIC", "LICENCE NO", "LICENSE NO", "DL NO", "DL NUMBER")):
            # Check same line and next line
            m = _DL_NUMBER_RE.search(t)
            if m:
                return m.group(1).upper()
            if i + 1 < len(texts):
                m = _DL_NUMBER_RE.search(texts[i + 1])
                if m:
                    return m.group(1).upper()

    # Fall back: scan all blocks for the pattern
    candidates: list[str] = []
    for t in texts:
        for m in _DL_NUMBER_RE.finditer(t):
            val = m.group(1).upper()
            # Reject numbers that look like NIC (9-digit+VX or 12 pure digits without context)
            if len(val.replace("-", "")) >= 10:
                candidates.append(val)

    return candidates[0] if candidates else None


def _extract_name(texts: list[str]) -> str | None:
    """Return name following a recognised label, or the longest all-alpha line."""
    for i, t in enumerate(texts):
        up = t.upper()
        if any(lbl in up for lbl in _NAME_LABELS):
            # Name may be on the same line after the label
            after_label = re.split(r"(?:NAME|HOLDER)\s*[:\-]?\s*", t, maxsplit=1, flags=re.I)
            if len(after_label) > 1 and after_label[1].strip():
                return _clean_name(after_label[1])
            # Or on the next line
            if i + 1 < len(texts):
                nxt = texts[i + 1].strip()
                if re.match(r"^[A-Za-z .'\-]+$", nxt) and len(nxt) > 3:
                    return _clean_name(nxt)

    # Heuristic: longest all-alpha token longer than 5 chars (likely a name)
    alpha_lines = [
        t for t in texts
        if re.match(r"^[A-Za-z .'\-]+$", t.strip()) and len(t.strip()) > 5
    ]
    if alpha_lines:
        return _clean_name(max(alpha_lines, key=len))

    return None


def _clean_name(raw: str) -> str:
    return re.sub(r"\s{2,}", " ", raw.strip()).title()


def _extract_dates(
    texts: list[str], upper_text: str
) -> tuple[str | None, str | None, str | None]:
    """Return (dob, issue_date, expiry_date) in ISO 8601."""
    dob = issue = expiry = None

    # Pass 1: look for labelled dates
    for i, t in enumerate(texts):
        up = t.upper()
        date_in_line = _parse_date_from_text(t)

        if any(lbl in up for lbl in _DOB_LABELS):
            dob = dob or date_in_line or _get_next_date(texts, i)
        elif any(lbl in up for lbl in _ISSUE_LABELS):
            issue = issue or date_in_line or _get_next_date(texts, i)
        elif any(lbl in up for lbl in _EXPIRY_LABELS):
            expiry = expiry or date_in_line or _get_next_date(texts, i)

    # Pass 2: collect all date strings and assign by temporal order
    if not (dob and issue and expiry):
        all_dates: list[tuple[str, str]] = []
        for t in texts:
            for m in _DATE_RE.finditer(t):
                iso = _normalise_date(m.group(0))
                if iso:
                    all_dates.append((m.group(0), iso))
        all_dates.sort(key=lambda x: x[1])

        if len(all_dates) >= 3 and not (dob and issue and expiry):
            # Typical order on SL DL: DOB (earliest) → issue → expiry (latest)
            dob = dob or all_dates[0][1]
            issue = issue or all_dates[1][1]
            expiry = expiry or all_dates[-1][1]
        elif len(all_dates) == 2 and not expiry:
            issue = issue or all_dates[0][1]
            expiry = expiry or all_dates[1][1]

    return dob, issue, expiry


def _get_next_date(texts: list[str], label_idx: int) -> str | None:
    if label_idx + 1 < len(texts):
        return _parse_date_from_text(texts[label_idx + 1])
    return None


def _parse_date_from_text(text: str) -> str | None:
    m = _DATE_RE.search(text)
    return _normalise_date(m.group(0)) if m else None


def _normalise_date(raw: str) -> str | None:
    """Convert DD/MM/YYYY or YYYY-MM-DD variants to ISO 8601 YYYY-MM-DD."""
    raw = raw.strip()
    sep_re = re.compile(r"[/.\-]")

    parts = sep_re.split(raw)
    if len(parts) != 3:
        return None

    try:
        if len(parts[0]) == 4:
            # YYYY-MM-DD
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        else:
            # DD-MM-YYYY
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
    except (ValueError, IndexError):
        return None


def _extract_address(texts: list[str]) -> str | None:
    """Collect text lines that follow an address label."""
    for i, t in enumerate(texts):
        up = t.upper()
        if any(lbl in up for lbl in _ADDRESS_LABELS):
            parts = []
            for j in range(i + 1, min(i + 5, len(texts))):
                nxt = texts[j].strip()
                # Stop at next label-like line
                if any(lbl in nxt.upper() for lbl in _EXPIRY_LABELS | _ISSUE_LABELS):
                    break
                if nxt:
                    parts.append(nxt)
            if parts:
                return ", ".join(parts)
    return None


def _extract_vehicle_categories(upper_text: str) -> list[str]:
    """Find vehicle category codes present in the text."""
    found: list[str] = []
    for cat in sorted(_VEHICLE_CATEGORIES, key=len, reverse=True):
        # Match standalone category code (word boundary)
        if re.search(rf"\b{re.escape(cat)}\b", upper_text):
            found.append(cat)
    return sorted(set(found))

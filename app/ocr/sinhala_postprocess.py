"""
Sinhala OCR post-processing utilities.

PaddleOCR (English model) sometimes recognises Sinhala unicode as garbled
ASCII sequences or confuses similar glyphs. This module provides:

  1. Unicode normalisation — NFC form, composing combining characters
  2. Common substitution-error correction for ArcFace-like glyph confusion
  3. Transliteration helpers for Sinhala → Latin (for field matching)
  4. Detection of whether a text block likely contains Sinhala script

Sinhala Unicode range: U+0D80 – U+0DFF
"""

from __future__ import annotations

import re
import unicodedata

# ── Sinhala unicode range ─────────────────────────────────────────────────────
_SINHALA_START = 0x0D80
_SINHALA_END = 0x0DFF

# Sinhala character categories as sets of codepoints
_SINHALA_VOWELS = set(range(0x0D85, 0x0D97))       # Independent vowels
_SINHALA_CONSONANTS = set(range(0x0D9A, 0x0DC7))   # Consonants
_SINHALA_VOWEL_MARKS = set(range(0x0DCF, 0x0DDF))  # Dependent vowel signs
_SINHALA_DIGITS = set(range(0x0DE6, 0x0DEF))        # Sinhala digits

# ── Common PaddleOCR glyph-confusion corrections (Sinhala context) ─────────
# Map sequences that OCR frequently produces to their correct Sinhala form.
# These are approximate corrections for common confusion pairs.
_SINHALA_CORRECTIONS: dict[str, str] = {
    # HAL + YANSAYA confusion
    "්‍ය": "්‍ය",  # ZWJ yansaya (already correct)
    # Common vowel sign confusions
    "ේ": "ේ",   # e-sign + hal → ē-sign
    "ො": "ෛ",   # e-sign + aa-sign → ō-sign (approximate)
    # Standalone virama that should be connecting
    " ් ": "්",       # space-hal-space → hal
}

# Latin lookalike corrections for OCR output containing Sinhala-adjacent text
_LATIN_CONFUSION: dict[str, str] = {
    "0": "O",   # zero → letter O (in name fields)
    "1": "I",   # one → letter I (in name fields)
    "|": "I",   # pipe → I
    "©": "O",   # copyright → O
}


# ── Public API ────────────────────────────────────────────────────────────────

def is_sinhala_text(text: str) -> bool:
    """Return True if the text contains any Sinhala script codepoints."""
    return any(_SINHALA_START <= ord(c) <= _SINHALA_END for c in text)


def contains_sinhala(blocks: list[dict]) -> bool:
    """Return True if any OCR block contains Sinhala characters."""
    return any(is_sinhala_text(b.get("text", "")) for b in blocks)


def normalise_sinhala(text: str) -> str:
    """
    NFC-normalise and apply common correction rules to Sinhala text.

    NFC composes base characters with their combining marks (vowel signs,
    HAL/YANSAYA) into canonical sequences that regex and field-matching
    can handle reliably.
    """
    text = unicodedata.normalize("NFC", text)
    for wrong, right in _SINHALA_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


def normalise_blocks(blocks: list[dict]) -> list[dict]:
    """
    Return a new list of blocks with Sinhala text normalised in-place.
    Non-Sinhala blocks are returned unchanged.
    """
    result = []
    for b in blocks:
        t = b.get("text", "")
        if is_sinhala_text(t):
            b = {**b, "text": normalise_sinhala(t)}
        result.append(b)
    return result


def split_bilingual_blocks(blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Partition OCR blocks into (english_blocks, sinhala_blocks).

    Blocks with mixed content are classified by the dominant script.
    """
    en_blocks: list[dict] = []
    si_blocks: list[dict] = []
    for b in blocks:
        t = b.get("text", "")
        si_chars = sum(1 for c in t if _SINHALA_START <= ord(c) <= _SINHALA_END)
        if si_chars / max(len(t), 1) > 0.3:
            si_blocks.append(b)
        else:
            en_blocks.append(b)
    return en_blocks, si_blocks


def clean_name_ocr(name: str, sinhala: bool = False) -> str:
    """
    Remove common OCR artefacts from a name field.

    For English names: fix digit/letter confusions, strip punctuation that
    doesn't belong in a name, normalise whitespace.
    For Sinhala names: NFC-normalise and strip non-Sinhala ASCII noise.
    """
    if sinhala:
        name = normalise_sinhala(name)
        # Strip ASCII characters that are clearly noise in Sinhala context
        name = re.sub(r"[^඀-෿\s]", "", name)
        return name.strip()

    # English: fix common digit/symbol → letter substitutions
    for digit, letter in _LATIN_CONFUSION.items():
        name = name.replace(digit, letter)
    # Strip non-name characters
    name = re.sub(r"[^A-Za-z .\-']", "", name)
    return re.sub(r"\s{2,}", " ", name).strip().title()


# ── Transliteration (Sinhala → Latin, approximate) ───────────────────────────
# Used for matching field values when English transliteration is provided
# alongside Sinhala text.

# Partial phoneme map — covers the most common consonants and vowels
_SI_TO_LATIN: dict[str, str] = {
    "අ": "a",   "ආ": "aa",  "ඇ": "ae",  "ඈ": "aee",
    "ඉ": "i",   "ඊ": "ii",  "උ": "u",   "ඌ": "uu",
    "එ": "e",   "ඒ": "ee",  "ඔ": "o",   "ඖ": "au",
    "ක": "k",   "ග": "g",   "ඟ": "ng",  "ච": "c",
    "ජ": "j",   "ඤ": "ny",  "ට": "t",   "ඩ": "d",
    "ත": "n",   "ප": "p",   "බ": "b",   "ම": "m",
    "ය": "y",   "ර": "r",   "ල": "l",   "ව": "v",
    "ස": "s",   "හ": "h",   "ළ": "l",   "ෆ": "f",
    "ා": "aa",  "ැ": "ae",  "ෑ": "aee", "ි": "i",
    "ී": "ii",  "ු": "u",   "ූ": "uu",  "ෙ": "e",
    "ේ": "ee",  "ෛ": "o",   "ො": "o",   "ෝ": "oo",
    "්": "",    # HAL (virama) — suppresses inherent vowel
}


def transliterate_sinhala(text: str) -> str:
    """Approximate Sinhala → Latin phonetic transliteration."""
    out = []
    for ch in text:
        out.append(_SI_TO_LATIN.get(ch, ch if ord(ch) < 128 else ""))
    return "".join(out)

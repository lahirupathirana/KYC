"""Tests for passport MRZ parsing and heuristic extraction."""

import pytest

from app.ocr.passport_parser import (
    PassportFields,
    _decode_old_nic,
    _normalise_date,
    _parse_mrz,
    _parse_mrz_date,
    _try_parse_pair,
    parse,
)


# ── MRZ date parsing ──────────────────────────────────────────────────────────

def test_mrz_date_valid_dob():
    # DOB: 850312 → 1985-03-12
    result = _parse_mrz_date("850312", is_dob=True)
    assert result == "1985-03-12"


def test_mrz_date_valid_dob_2000s():
    # DOB: 010501 → 2001-05-01 (01 ≤ current year short)
    result = _parse_mrz_date("010501", is_dob=True)
    assert result == "2001-05-01"


def test_mrz_date_valid_expiry():
    result = _parse_mrz_date("301231", is_dob=False)
    assert result == "2030-12-31"


def test_mrz_date_invalid_month():
    assert _parse_mrz_date("851332", is_dob=True) is None


def test_mrz_date_non_digits():
    assert _parse_mrz_date("8503AB", is_dob=True) is None


# ── MRZ line pair parsing ─────────────────────────────────────────────────────

def test_try_parse_pair_valid_passport():
    # Synthetic TD3 MRZ
    line1 = "P<LKAMENDIS<<NIROSHAN<KUMARA<<<<<<<<<<<<<<<<<"
    line2 = "A1234567<8LKA8809014M2512317<<<<<<<<<<<<<<4"
    # Pad to 44 chars
    line1 = line1[:44].ljust(44, "<")
    line2 = line2[:44].ljust(44, "<")
    result = _try_parse_pair(line1, line2)
    assert result is not None
    assert result.surname == "MENDIS"
    assert "NIROSHAN" in result.given_names
    assert result.dob == "1988-09-01"
    assert result.document_number == "A1234567"
    assert result.country_code == "LKA"


def test_try_parse_pair_invalid_first_char():
    line1 = "X<LKASURNAME<<FIRSTNAME<<<<<<<<<<<<<<<<<<<<<<"
    line2 = "A1234567<8LKA8809014M2512317<<<<<<<<<<<<<<4"
    line1 = line1[:44].ljust(44, "<")
    line2 = line2[:44].ljust(44, "<")
    # X is not a valid passport type indicator for TD3 (P is standard)
    # Parser may still try — just verify it doesn't crash
    result = _try_parse_pair(line1, line2)
    # May be None or partial — just no exception
    assert result is None or isinstance(result, PassportFields)


def test_parse_mrz_finds_lines_in_block_list():
    line1 = "P<LKAPERERA<<KAMALA<DEVI<<<<<<<<<<<<<<<<<<<<".ljust(44, "<")
    line2 = "B9876543<2LKA9203156F2703311<<<<<<<<<<<<<<2".ljust(44, "<")
    texts = ["SRI LANKA", "PASSPORT", line1, line2, "SOME OTHER TEXT"]
    result = _parse_mrz(texts)
    assert result is not None
    assert result.surname is not None


# ── Heuristic fallback ────────────────────────────────────────────────────────

def test_parse_heuristic_extracts_passport_number():
    blocks = [
        {"text": "REPUBLIC OF SRI LANKA", "confidence": 0.95},
        {"text": "PASSPORT", "confidence": 0.95},
        {"text": "Passport No", "confidence": 0.9},
        {"text": "N1234567", "confidence": 0.9},
        {"text": "Date of Birth", "confidence": 0.9},
        {"text": "15/03/1990", "confidence": 0.85},
        {"text": "Name", "confidence": 0.9},
        {"text": "Kamal Perera", "confidence": 0.85},
    ]
    result = parse(blocks)
    assert result is not None
    assert result.document_number == "N1234567"
    assert result.dob == "1990-03-15"


# ── Date normalisation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("15/03/1990", "1990-03-15"),
    ("15-03-1990", "1990-03-15"),
    ("15.03.1990", "1990-03-15"),
    ("19900315",   "1990-03-15"),
])
def test_normalise_date(raw, expected):
    assert _normalise_date(raw) == expected


def test_normalise_date_invalid():
    assert _normalise_date("not-a-date") is None

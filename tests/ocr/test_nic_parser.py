"""Tests for Sri Lankan NIC parsing and DOB decoding."""

import pytest

from app.ocr.nic_parser import (
    NICFields,
    _decode_new_nic,
    _decode_old_nic,
    _extract_nic_number,
    parse,
)


# ── Old NIC decoding ──────────────────────────────────────────────────────────

def test_old_nic_male_dob():
    # 890123456V → 1989, day 012 → Jan 12
    dob, sex = _decode_old_nic("890123456")
    assert dob == "1989-01-12"
    assert sex == "M"


def test_old_nic_female_dob():
    # 890623456V → 1989, day 062+500=562 female → day 62 → Mar 3
    dob, sex = _decode_old_nic("890623456")
    assert sex == "F"
    # day 062: Jan(31) + Feb(28) + 3 = Mar 3 in 1989 (non-leap)
    assert dob == "1989-03-03"


def test_old_nic_leap_year():
    # 960601234V → 1996 (leap), day 060 → Feb 29
    dob, sex = _decode_old_nic("960601234")
    assert sex == "M"
    assert dob == "1996-02-29"


# ── New NIC decoding ──────────────────────────────────────────────────────────

def test_new_nic_male_dob():
    # 199801200123 → 1998, day 012 → Jan 12, male
    dob, sex = _decode_new_nic("199801200123")
    assert dob == "1998-01-12"
    assert sex == "M"


def test_new_nic_female_dob():
    # 200051201234 → 2000, day 512 → 512-500=12 → Jan 12, female
    dob, sex = _decode_new_nic("200051201234")
    assert dob == "2000-01-12"
    assert sex == "F"


# ── NIC number extraction from text ──────────────────────────────────────────

def test_extract_old_nic_from_text():
    nic_no, dob, sex = _extract_nic_number("NIC: 890123456V some other text")
    assert nic_no == "890123456V"
    assert dob is not None
    assert sex == "M"


def test_extract_new_nic_from_text():
    nic_no, dob, sex = _extract_nic_number("ID Number 199801200123 issued")
    assert nic_no == "199801200123"
    assert dob == "1998-01-12"


def test_extract_nic_returns_none_on_no_match():
    nic, dob, sex = _extract_nic_number("No NIC here")
    assert nic is None
    assert dob is None


# ── Full parse ────────────────────────────────────────────────────────────────

def test_parse_full_old_nic_card():
    blocks = [
        {"text": "DEMOCRATIC SOCIALIST REPUBLIC OF SRI LANKA", "confidence": 0.95},
        {"text": "NATIONAL IDENTITY CARD", "confidence": 0.95},
        {"text": "NAME", "confidence": 0.9},
        {"text": "KAMAL BANDARA PERERA", "confidence": 0.88},
        {"text": "NIC No", "confidence": 0.9},
        {"text": "890123456V", "confidence": 0.92},
    ]
    result = parse(blocks)
    assert result is not None
    assert result.document_number == "890123456V"
    assert result.dob == "1989-01-12"
    assert result.sex == "M"
    assert result.full_name is not None
    assert result.confidence > 0.5


def test_parse_returns_none_when_nothing_found():
    blocks = [{"text": "Hello World", "confidence": 0.9}]
    result = parse(blocks)
    assert result is None


def test_parse_confidence_proportional_to_fields():
    # Only NIC number found → 1 of 3 fields
    blocks = [{"text": "990234567V", "confidence": 0.9}]
    result = parse(blocks)
    assert result is not None
    assert result.confidence == pytest.approx(1 / 3, abs=0.01)

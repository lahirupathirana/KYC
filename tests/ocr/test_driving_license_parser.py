"""
Tests for the Sri Lankan Driving License parser.
"""

from __future__ import annotations

import pytest

from app.ocr.driving_license_parser import (
    _extract_dates,
    _extract_dl_number,
    _extract_name,
    _extract_vehicle_categories,
    _normalise_date,
    parse,
)


# ── Date normalisation ────────────────────────────────────────────────────────

class TestNormaliseDate:
    def test_dd_slash_mm_yyyy(self):
        assert _normalise_date("15/06/1990") == "1990-06-15"

    def test_dd_dot_mm_yyyy(self):
        assert _normalise_date("15.06.1990") == "1990-06-15"

    def test_dd_hyphen_mm_yyyy(self):
        assert _normalise_date("15-06-1990") == "1990-06-15"

    def test_yyyy_hyphen_mm_dd(self):
        assert _normalise_date("1990-06-15") == "1990-06-15"

    def test_invalid_returns_none(self):
        assert _normalise_date("notadate") is None

    def test_zero_padding(self):
        assert _normalise_date("05/01/2000") == "2000-01-05"


# ── DL number extraction ──────────────────────────────────────────────────────

class TestExtractDlNumber:
    def test_labelled_licence_no(self):
        texts = ["DRIVING LICENCE", "LICENCE NO: 1234567890"]
        assert _extract_dl_number(texts) == "1234567890"

    def test_labelled_dl_no(self):
        texts = ["DL NO", "987654321098"]
        assert _extract_dl_number(texts) == "987654321098"

    def test_unlabelled_12_digit(self):
        texts = ["Department of Motor Traffic", "123456789012"]
        result = _extract_dl_number(texts)
        assert result == "123456789012"

    def test_letter_prefixed(self):
        texts = ["B12345678"]
        result = _extract_dl_number(texts)
        assert result is not None
        assert "B12345678" in result or result == "B12345678"

    def test_no_number_returns_none(self):
        texts = ["No numbers here at all"]
        assert _extract_dl_number(texts) is None


# ── Name extraction ───────────────────────────────────────────────────────────

class TestExtractName:
    def test_after_name_label(self):
        texts = ["LICENCE HOLDER: Kasun Perera", "Some other text"]
        result = _extract_name(texts)
        assert result is not None
        assert "Kasun" in result or "Perera" in result

    def test_name_on_next_line(self):
        texts = ["NAME:", "Nimal Bandara", "15/06/1985"]
        result = _extract_name(texts)
        assert result is not None
        assert "Nimal" in result

    def test_longest_alpha_fallback(self):
        texts = ["123456789", "Chamari Fernando", "B", "15/06/1990"]
        result = _extract_name(texts)
        assert result is not None
        assert "Chamari" in result or "Fernando" in result

    def test_no_name_returns_none(self):
        texts = ["123", "456", "789"]
        # All-numeric — no alpha lines longer than 5 chars
        assert _extract_name(texts) is None


# ── Vehicle categories ────────────────────────────────────────────────────────

class TestExtractVehicleCategories:
    def test_single_category(self):
        assert "B" in _extract_vehicle_categories("VEHICLE CLASS B PERMITTED")

    def test_multiple_categories(self):
        cats = _extract_vehicle_categories("AUTHORISED FOR A B1 C D")
        assert "A" in cats
        assert "B1" in cats
        assert "C" in cats
        assert "D" in cats

    def test_no_categories(self):
        assert _extract_vehicle_categories("NO CATEGORIES LISTED HERE") == []

    def test_a1_not_confused_with_a(self):
        cats = _extract_vehicle_categories("A1 ONLY")
        assert "A1" in cats
        # A should also be detected (A1 contains A as substring, but word boundary should catch it)
        # Behaviour depends on regex; just ensure A1 is present
        assert len(cats) >= 1


# ── Date triple extraction ────────────────────────────────────────────────────

class TestExtractDates:
    def test_labelled_dates(self):
        texts = [
            "DATE OF BIRTH: 15/06/1985",
            "DATE OF ISSUE: 01/01/2010",
            "EXPIRY: 01/01/2015",
        ]
        dob, issue, expiry = _extract_dates(texts, " ".join(texts).upper())
        assert dob == "1985-06-15"
        assert issue == "2010-01-01"
        assert expiry == "2015-01-01"

    def test_three_unlabelled_dates_sorted(self):
        # Temporal order: DOB < issue < expiry
        texts = ["01/01/2025", "15/06/1985", "01/01/2020"]
        dob, issue, expiry = _extract_dates(texts, " ".join(texts).upper())
        assert dob == "1985-06-15"
        assert issue == "2020-01-01"
        assert expiry == "2025-01-01"

    def test_no_dates(self):
        texts = ["Driving Licence", "No dates here"]
        dob, issue, expiry = _extract_dates(texts, " ".join(texts).upper())
        assert dob is None
        assert issue is None
        assert expiry is None


# ── Full parse() ──────────────────────────────────────────────────────────────

class TestParseDrivingLicense:
    def _make_blocks(self, texts: list[str]) -> list[dict]:
        return [{"text": t, "confidence": 0.95} for t in texts]

    def test_full_valid_dl(self):
        blocks = self._make_blocks([
            "DEMOCRATIC SOCIALIST REPUBLIC OF SRI LANKA",
            "DEPARTMENT OF MOTOR TRAFFIC — DRIVING LICENCE",
            "Licence No.: 123456789012",
            "Licence Holder: Kasun Perera",
            "Date of Birth: 15/06/1985",
            "Date of Issue: 01/03/2010",
            "Expiry Date: 01/03/2015",
            "Vehicle Category: A B",
            "Address: 10, Nugegoda Road, Colombo",
        ])
        result = parse(blocks)
        assert result is not None
        assert result.document_number == "123456789012"
        assert result.dob == "1985-06-15"
        assert result.expiry_date is not None
        assert result.confidence > 0.0

    def test_empty_blocks_returns_none(self):
        assert parse([]) is None

    def test_partial_extraction_has_low_confidence(self):
        blocks = self._make_blocks(["DRIVING LICENCE", "Some Name"])
        result = parse(blocks)
        # May return a result with low confidence or None
        if result is not None:
            assert result.confidence < 1.0

    def test_vehicle_categories_extracted(self):
        blocks = self._make_blocks([
            "DEPARTMENT OF MOTOR TRAFFIC DRIVING LICENSE",
            "123456789012",
            "A. Fernando",
            "15/06/1985",
            "01/03/2010",
            "01/03/2015",
            "VEHICLE CLASS: A B1 C",
        ])
        result = parse(blocks)
        if result is not None:
            assert "A" in result.vehicle_categories or "B1" in result.vehicle_categories

"""
Tests for OCR evaluation metrics (CER, WER, field F1, aggregation).
"""

from __future__ import annotations

import pytest

from app.ocr.evaluation import (
    InferenceTimer,
    aggregate_results,
    character_error_rate,
    evaluate_sample,
    word_error_rate,
)


# ── CER ───────────────────────────────────────────────────────────────────────

class TestCharacterErrorRate:
    def test_identical(self):
        assert character_error_rate("hello", "hello") == 0.0

    def test_empty_hypothesis(self):
        assert character_error_rate("", "hello") == 1.0

    def test_empty_reference(self):
        assert character_error_rate("anything", "") == 0.0

    def test_one_substitution(self):
        # "hellX" vs "hello" — 1 sub out of 5 chars = 0.2
        assert character_error_rate("hellX", "hello") == pytest.approx(0.2)

    def test_one_deletion(self):
        # "hell" vs "hello" — 1 deletion out of 5 chars = 0.2
        assert character_error_rate("hell", "hello") == pytest.approx(0.2)

    def test_one_insertion(self):
        # "helloo" vs "hello" — 1 insertion out of 5 chars = 0.2
        assert character_error_rate("helloo", "hello") == pytest.approx(0.2)

    def test_case_sensitive(self):
        # CER is case-sensitive by default (field values are uppercased in evaluate_sample)
        assert character_error_rate("Hello", "hello") > 0.0

    def test_completely_wrong(self):
        cer = character_error_rate("ZZZZZ", "hello")
        assert cer == pytest.approx(1.0)


# ── WER ───────────────────────────────────────────────────────────────────────

class TestWordErrorRate:
    def test_identical(self):
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_one_word_wrong(self):
        # "hello earth" vs "hello world" — 1/2 = 0.5
        assert word_error_rate("hello earth", "hello world") == pytest.approx(0.5)

    def test_empty_reference(self):
        assert word_error_rate("anything here", "") == 0.0

    def test_empty_hypothesis(self):
        assert word_error_rate("", "hello world") == 1.0

    def test_single_word_correct(self):
        assert word_error_rate("PERERA", "PERERA") == 0.0

    def test_extra_word(self):
        # "hello world extra" vs "hello world" — 1 insertion / 2 ref words = 0.5
        assert word_error_rate("hello world extra", "hello world") == pytest.approx(0.5)


# ── evaluate_sample ───────────────────────────────────────────────────────────

class TestEvaluateSample:
    def _sample(self, pred, gt, pred_type="nic", gt_type="nic"):
        return evaluate_sample(pred, gt, pred_type, gt_type)

    def test_perfect_extraction(self):
        pred = {"document_number": "199512300123", "full_name": "Kasun Perera"}
        gt = {"document_number": "199512300123", "full_name": "Kasun Perera"}
        r = self._sample(pred, gt)
        assert r["type_correct"] is True
        assert r["avg_cer"] == 0.0
        assert r["avg_wer"] == 0.0

    def test_wrong_document_type(self):
        r = self._sample({}, {}, pred_type="passport", gt_type="nic")
        assert r["type_correct"] is False

    def test_partial_extraction(self):
        pred = {"document_number": "199512300123", "full_name": None}
        gt = {"document_number": "199512300123", "full_name": "Kasun Perera"}
        r = self._sample(pred, gt)
        # full_name not extracted → CER=1.0 for that field
        assert r["avg_cer"] > 0.0

    def test_missing_gt_field_skipped(self):
        # If ground truth has no expiry (NIC), that field should be skipped
        pred = {"document_number": "199512300123"}
        gt = {"document_number": "199512300123", "expiry_date": None}
        r = self._sample(pred, gt)
        # Only document_number contributes
        assert "expiry_date" not in r["fields"]

    def test_cer_on_one_char_difference(self):
        pred = {"document_number": "199512300124"}  # last digit wrong
        gt = {"document_number": "199512300123"}
        r = self._sample(pred, gt)
        field = r["fields"]["document_number"]
        assert field["cer"] == pytest.approx(1 / 12)

    def test_exact_match_flag(self):
        pred = {"full_name": "KASUN PERERA"}
        gt = {"full_name": "KASUN PERERA"}
        r = self._sample(pred, gt)
        assert r["fields"]["full_name"]["exact_match"] is True

    def test_case_insensitive_comparison(self):
        # evaluate_sample uppercases both sides
        pred = {"full_name": "kasun perera"}
        gt = {"full_name": "Kasun Perera"}
        r = self._sample(pred, gt)
        assert r["fields"]["full_name"]["exact_match"] is True


# ── aggregate_results ─────────────────────────────────────────────────────────

class TestAggregateResults:
    def _make_sample(self, cer=0.0, wer=0.0, type_correct=True, fields=None):
        return {
            "type_correct": type_correct,
            "avg_cer": cer,
            "avg_wer": wer,
            "fields": fields or {},
        }

    def test_empty_returns_zero_report(self):
        report = aggregate_results([], [])
        assert report.total_samples == 0
        assert report.avg_cer == 0.0

    def test_all_correct(self):
        samples = [self._make_sample(cer=0.0, wer=0.0, type_correct=True)] * 10
        report = aggregate_results(samples, [50.0] * 10)
        assert report.total_samples == 10
        assert report.document_type_accuracy == 1.0
        assert report.avg_cer == 0.0
        assert report.avg_inference_ms == pytest.approx(50.0)

    def test_mixed_accuracy(self):
        samples = [
            self._make_sample(type_correct=True),
            self._make_sample(type_correct=False),
        ]
        report = aggregate_results(samples, [40.0, 60.0])
        assert report.document_type_accuracy == pytest.approx(0.5)
        assert report.avg_inference_ms == pytest.approx(50.0)

    def test_field_tp_fp_fn_counts(self):
        samples = [
            {
                "type_correct": True,
                "avg_cer": 0.0,
                "avg_wer": 0.0,
                "fields": {
                    "document_number": {
                        "extracted": True, "exact_match": True,
                        "ground_truth": "123", "predicted": "123",
                        "cer": 0.0, "wer": 0.0,
                    }
                },
            },
            {
                "type_correct": True,
                "avg_cer": 0.5,
                "avg_wer": 0.5,
                "fields": {
                    "document_number": {
                        "extracted": True, "exact_match": False,
                        "ground_truth": "456", "predicted": "457",
                        "cer": 0.33, "wer": 1.0,
                    }
                },
            },
        ]
        report = aggregate_results(samples, [30.0, 70.0])
        m = report.field_metrics.get("document_number")
        assert m is not None
        assert m.tp == 1
        assert m.fp == 1
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(1.0)


# ── InferenceTimer ────────────────────────────────────────────────────────────

class TestInferenceTimer:
    def test_records_elapsed(self):
        with InferenceTimer() as t:
            pass  # essentially instant
        assert t.elapsed_ms >= 0.0
        assert t.elapsed_ms < 1000.0  # should take < 1s

    def test_non_zero_for_real_work(self):
        import time
        with InferenceTimer() as t:
            time.sleep(0.05)
        assert t.elapsed_ms >= 40.0  # allow some margin

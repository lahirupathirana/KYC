"""
OCR evaluation metrics for research reporting.

Reports:
  - CER  (Character Error Rate)  — edit distance at character level
  - WER  (Word Error Rate)       — edit distance at word level
  - Precision / Recall / F1      — field-extraction accuracy
  - Average inference latency    — wall-clock timing

All functions are pure and have no external dependencies beyond the standard
library (edit distance uses a hand-rolled DP table to avoid requiring
`nltk` or `jiwer` in the research environment).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any


# ── Levenshtein edit distance ─────────────────────────────────────────────────

def _edit_distance(a: list, b: list) -> int:
    """Standard Levenshtein distance between two sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ── CER / WER ─────────────────────────────────────────────────────────────────

def character_error_rate(hypothesis: str, reference: str) -> float:
    """
    CER = edit_distance(chars(hyp), chars(ref)) / len(chars(ref))

    Returns 0.0 when reference is empty (avoids division-by-zero).
    May exceed 1.0 when hypothesis is much longer than reference.
    """
    if not reference:
        return 0.0
    return _edit_distance(list(hypothesis), list(reference)) / len(reference)


def word_error_rate(hypothesis: str, reference: str) -> float:
    """
    WER = edit_distance(words(hyp), words(ref)) / len(words(ref))

    Tokenises on whitespace; returns 0.0 on empty reference.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0
    return _edit_distance(hyp_words, ref_words) / len(ref_words)


# ── Field-extraction precision / recall / F1 ─────────────────────────────────

@dataclass
class FieldMetrics:
    field_name: str
    tp: int = 0     # true positives  (extracted, matches ground truth)
    fp: int = 0     # false positives (extracted, wrong value)
    fn: int = 0     # false negatives (not extracted, ground truth exists)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvaluationReport:
    """Aggregate evaluation results over a dataset split."""
    total_samples: int = 0
    document_type_accuracy: float = 0.0
    avg_cer: float = 0.0
    avg_wer: float = 0.0
    avg_inference_ms: float = 0.0
    std_inference_ms: float = 0.0
    field_metrics: dict[str, FieldMetrics] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "document_type_accuracy": round(self.document_type_accuracy, 4),
            "avg_cer": round(self.avg_cer, 4),
            "avg_wer": round(self.avg_wer, 4),
            "avg_inference_ms": round(self.avg_inference_ms, 1),
            "std_inference_ms": round(self.std_inference_ms, 1),
            "fields": {
                name: {
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1": round(m.f1, 4),
                }
                for name, m in self.field_metrics.items()
            },
        }


# ── Per-sample evaluation ─────────────────────────────────────────────────────

CORE_FIELDS = ("document_number", "full_name", "dob", "expiry_date")


def evaluate_sample(
    predicted: dict[str, str | None],
    ground_truth: dict[str, str | None],
    predicted_doc_type: str,
    ground_truth_doc_type: str,
) -> dict[str, Any]:
    """
    Compare one prediction against ground truth.

    Args:
        predicted: extracted field values (None = not extracted)
        ground_truth: reference field values (None = field not present in doc)
        predicted_doc_type: e.g. "nic", "passport", "driving_license"
        ground_truth_doc_type: reference document type

    Returns dict with per-field cer/wer, precision/recall, type_correct flag.
    """
    type_correct = predicted_doc_type.lower() == ground_truth_doc_type.lower()

    results: dict[str, Any] = {"type_correct": type_correct, "fields": {}}
    cers, wers = [], []

    for f in CORE_FIELDS:
        pred_val = (predicted.get(f) or "").strip().upper()
        gt_val = (ground_truth.get(f) or "").strip().upper()

        if not gt_val:
            continue  # field not expected for this document type

        cer = character_error_rate(pred_val, gt_val)
        wer = word_error_rate(pred_val, gt_val)
        cers.append(cer)
        wers.append(wer)

        extracted = pred_val != ""
        exact_match = pred_val == gt_val

        results["fields"][f] = {
            "predicted": pred_val or None,
            "ground_truth": gt_val,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact_match": exact_match,
            "extracted": extracted,
        }

    results["avg_cer"] = round(mean(cers), 4) if cers else 1.0
    results["avg_wer"] = round(mean(wers), 4) if wers else 1.0
    return results


# ── Dataset-level aggregation ─────────────────────────────────────────────────

def aggregate_results(
    sample_results: list[dict[str, Any]],
    inference_times_ms: list[float],
) -> EvaluationReport:
    """
    Aggregate per-sample evaluate_sample() outputs into an EvaluationReport.

    Args:
        sample_results: list of dicts from evaluate_sample()
        inference_times_ms: wall-clock inference times per sample (milliseconds)
    """
    n = len(sample_results)
    if n == 0:
        return EvaluationReport()

    type_correct = sum(1 for r in sample_results if r["type_correct"]) / n
    all_cer = [r["avg_cer"] for r in sample_results]
    all_wer = [r["avg_wer"] for r in sample_results]

    field_metrics: dict[str, FieldMetrics] = {f: FieldMetrics(f) for f in CORE_FIELDS}

    for r in sample_results:
        for fname, fdata in r["fields"].items():
            if fname not in field_metrics:
                field_metrics[fname] = FieldMetrics(fname)
            m = field_metrics[fname]
            if fdata["extracted"] and fdata["exact_match"]:
                m.tp += 1
            elif fdata["extracted"] and not fdata["exact_match"]:
                m.fp += 1
            elif not fdata["extracted"]:
                m.fn += 1

    return EvaluationReport(
        total_samples=n,
        document_type_accuracy=round(type_correct, 4),
        avg_cer=round(mean(all_cer), 4),
        avg_wer=round(mean(all_wer), 4),
        avg_inference_ms=round(mean(inference_times_ms), 1) if inference_times_ms else 0.0,
        std_inference_ms=round(stdev(inference_times_ms), 1) if len(inference_times_ms) > 1 else 0.0,
        field_metrics={k: v for k, v in field_metrics.items() if v.tp + v.fp + v.fn > 0},
    )


# ── Timing context manager ────────────────────────────────────────────────────

class InferenceTimer:
    """Context manager that records wall-clock time in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "InferenceTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000

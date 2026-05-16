"""
Batch OCR evaluation script.

Runs the KYC AI service OCR pipeline against a ground-truth manifest and
reports CER, WER, field-level F1, and inference latency.

Usage:
    # Start the dev server first:
    #   $env:ENABLED_MODELS='["ocr"]'; uvicorn app.main:app --port 8000
    #
    python scripts/evaluate_ocr.py \\
        --images datasets/synthetic/nic \\
        --ground-truth datasets/synthetic/nic/manifest.json \\
        --api http://localhost:8000/api/v1/ocr/extract \\
        --doc-type nic \\
        --limit 100 \\
        --output results/eval_nic.json

Or run without a live server (uses local pipeline directly):
    python scripts/evaluate_ocr.py \\
        --images datasets/synthetic/nic \\
        --ground-truth datasets/synthetic/nic/manifest.json \\
        --doc-type nic \\
        --local

Requires:
    pip install httpx  (for API mode)
    pip install paddleocr paddlepaddle  (for --local mode)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ocr.evaluation import (
    EvaluationReport,
    InferenceTimer,
    aggregate_results,
    evaluate_sample,
)


def _run_via_api(
    img_path: Path,
    api_url: str,
    doc_type: str | None,
) -> tuple[dict, float]:
    """POST image to OCR API endpoint; return (fields_dict, elapsed_ms)."""
    try:
        import httpx
    except ImportError:
        print("httpx not installed. Run: pip install httpx", file=sys.stderr)
        sys.exit(1)

    params = {}
    if doc_type:
        params["doc_type"] = doc_type

    with InferenceTimer() as t:
        with open(img_path, "rb") as f:
            response = httpx.post(
                api_url,
                files={"file": (img_path.name, f, "image/jpeg")},
                params=params,
                timeout=60.0,
            )
    response.raise_for_status()
    data = response.json()
    fields = data.get("fields", {})
    pred_type = data.get("document_type", "unknown")
    return {**fields, "_doc_type": pred_type}, t.elapsed_ms


def _run_local(
    img_path: Path,
    doc_type: str | None,
    model,
) -> tuple[dict, float]:
    """Run the pipeline in-process without a running server."""
    from app.ocr.pipeline import run_pipeline

    image_bytes = img_path.read_bytes()

    from app.schemas.ocr import DocumentType as DT
    hint = None
    if doc_type:
        try:
            hint = DT(doc_type)
        except ValueError:
            pass

    with InferenceTimer() as t:
        result = run_pipeline(image_bytes, model, doc_type_hint=hint)

    fields = result.fields.model_dump()
    fields["_doc_type"] = result.document_type.value
    return fields, t.elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch OCR evaluation")
    parser.add_argument("--images", required=True, help="Directory of test images")
    parser.add_argument("--ground-truth", required=True,
                        help="Path to manifest.json with ground truth")
    parser.add_argument("--api", default="http://localhost:8000/api/v1/ocr/extract",
                        help="OCR API URL (used unless --local is set)")
    parser.add_argument("--local", action="store_true",
                        help="Run pipeline in-process (requires paddleocr)")
    parser.add_argument("--doc-type", default=None,
                        help="Force document type hint: nic, passport, driving_license")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--output", default=None,
                        help="JSON file to write the evaluation report to")
    args = parser.parse_args()

    images_dir = Path(args.images)
    manifest = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))

    if args.limit:
        manifest = manifest[: args.limit]

    model = None
    if args.local:
        print("Loading PaddleOCR model…")
        from paddleocr import PaddleOCR
        model = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    sample_results: list[dict] = []
    times: list[float] = []
    errors = 0

    print(f"Evaluating {len(manifest)} samples…")

    for i, ann in enumerate(manifest):
        doc_type_gt = ann.get("document_type", "unknown")
        img_id = ann.get("id", i)
        img_name = f"{doc_type_gt}_{img_id:05d}.jpg"
        img_path = images_dir / img_name

        if not img_path.exists():
            # Try alternative naming
            img_path = next(images_dir.glob(f"*_{img_id:05d}.jpg"), None)

        if img_path is None or not img_path.exists():
            errors += 1
            continue

        try:
            if args.local:
                fields, elapsed = _run_local(img_path, args.doc_type, model)
            else:
                fields, elapsed = _run_via_api(img_path, args.api, args.doc_type)
        except Exception as exc:
            print(f"  ERROR on {img_path.name}: {exc}", file=sys.stderr)
            errors += 1
            continue

        gt = ann.get("ground_truth", {})
        pred_type = fields.pop("_doc_type", args.doc_type or "unknown")

        result = evaluate_sample(
            predicted=fields,
            ground_truth=gt,
            predicted_doc_type=pred_type,
            ground_truth_doc_type=doc_type_gt,
        )
        sample_results.append(result)
        times.append(elapsed)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(manifest)} — errors: {errors}")

    if not sample_results:
        print("No results collected. Check --images path and server availability.")
        sys.exit(1)

    report: EvaluationReport = aggregate_results(sample_results, times)
    summary = report.summary()
    summary["errors"] = errors

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total evaluated:         {report.total_samples}")
    print(f"  Errors/skipped:          {errors}")
    print(f"  Doc-type accuracy:       {report.document_type_accuracy:.2%}")
    print(f"  Avg CER:                 {report.avg_cer:.4f}")
    print(f"  Avg WER:                 {report.avg_wer:.4f}")
    print(f"  Avg inference latency:   {report.avg_inference_ms:.1f} ms")
    print(f"  Std inference latency:   {report.std_inference_ms:.1f} ms")
    print()
    print("  Field-level F1 scores:")
    for fname, m in report.field_metrics.items():
        print(f"    {fname:<25} P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()

import structlog

from app.schemas.scoring import (
    ComponentScore,
    ScoringInput,
    ScoringResult,
    VerificationDecision,
)

logger = structlog.get_logger()

# Relative importance of each verification signal.
# Weights sum to 1.0; missing components are re-normalized.
_WEIGHTS: dict[str, float] = {
    "ocr": 0.25,
    "face_match": 0.35,
    "liveness": 0.30,
    "voice": 0.10,
}

_APPROVED_THRESHOLD = 0.75   # high-confidence auto-approve
_REVIEW_THRESHOLD = 0.55     # borderline — route to human agent


class ScoringService:
    """
    Multi-modal decision engine — the core research contribution.

    Each pipeline component contributes a normalized score. The final score
    is an interpretable weighted sum, making the decision fully auditable
    for the MSc research evaluation.
    """

    async def compute_score(self, inputs: ScoringInput) -> ScoringResult:
        raw_scores: dict[str, float | None] = {
            "ocr": inputs.ocr_confidence,
            "face_match": inputs.face_similarity,
            "liveness": inputs.liveness_confidence,
            "voice": inputs.voice_confidence,
        }

        present = {k: v for k, v in raw_scores.items() if v is not None}
        if not present:
            raise ValueError("At least one scoring component must be provided")

        # Re-normalize weights so they sum to 1.0 even when components are missing
        weight_total = sum(_WEIGHTS[k] for k in present)
        components: list[ComponentScore] = []

        for key, raw in present.items():
            w = _WEIGHTS[key] / weight_total
            contribution = raw * w
            components.append(
                ComponentScore(
                    component=key,
                    raw_score=round(raw, 4),
                    weight=round(w, 4),
                    contribution=round(contribution, 4),
                    explanation=self._explain(key, raw),
                )
            )

        final_score = round(sum(c.contribution for c in components), 4)
        decision = self._decide(final_score)

        logger.info(
            "KYC score computed",
            session_id=inputs.session_id,
            score=final_score,
            decision=decision.value,
            components=[c.component for c in components],
        )

        return ScoringResult(
            session_id=inputs.session_id,
            final_score=final_score,
            decision=decision,
            components=components,
            threshold=_APPROVED_THRESHOLD,
            explanation=self._global_explanation(final_score, decision, components),
        )

    def _decide(self, score: float) -> VerificationDecision:
        if score >= _APPROVED_THRESHOLD:
            return VerificationDecision.APPROVED
        if score >= _REVIEW_THRESHOLD:
            return VerificationDecision.REVIEW
        return VerificationDecision.REJECTED

    def _explain(self, component: str, score: float) -> str:
        labels = {
            "ocr": f"Document text extracted with {score:.0%} average confidence",
            "face_match": f"Face-to-ID cosine similarity: {score:.4f}",
            "liveness": f"Liveness detection confidence: {score:.0%}",
            "voice": f"Voice response transcription confidence: {score:.0%}",
        }
        return labels.get(component, f"{component}: {score:.4f}")

    def _global_explanation(
        self,
        score: float,
        decision: VerificationDecision,
        components: list[ComponentScore],
    ) -> str:
        strongest = max(components, key=lambda c: c.contribution)
        weakest = min(components, key=lambda c: c.contribution)
        parts = [f"Final score {score} → {decision.value.upper()}."]
        parts.append(f"Strongest signal: {strongest.component} (contribution {strongest.contribution}).")
        if weakest.component != strongest.component:
            parts.append(f"Weakest signal: {weakest.component} (contribution {weakest.contribution}).")
        return " ".join(parts)

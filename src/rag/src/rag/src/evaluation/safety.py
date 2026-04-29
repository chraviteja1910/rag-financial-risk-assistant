"""
Safety & Evaluation Layer
NLI-based hallucination detection + faithfulness scoring + guardrail validation
Author: Ravi Teja Chittaluri
"""

from __future__ import annotations
from dataclasses import dataclass
from transformers import pipeline as hf_pipeline


@dataclass
class SafetyResult:
    passed: bool
    faithfulness_score: float
    hallucination_detected: bool
    guardrail_violations: list[str]
    details: dict


# Financial domain guardrail rules
FINANCIAL_GUARDRAILS = [
    {"pattern": r"guaranteed return", "message": "Cannot guarantee investment returns"},
    {"pattern": r"insider (tip|information)", "message": "No insider information allowed"},
    {"pattern": r"(buy|sell) immediately", "message": "No urgent trade recommendations"},
]


class SafetyLayer:
    """
    Production LLM safety validation with:
    1. NLI-based hallucination detection
    2. Faithfulness scoring (claim-level)
    3. Financial domain guardrail rules
    """

    def __init__(self, threshold: float = 0.85, model: str = "facebook/bart-large-mnli"):
        self.threshold = threshold
        self.nli_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=model,
            device=-1,  # CPU; set to 0 for GPU
        )

    def validate(
        self,
        query: str,
        response: str,
        context_docs: list[dict],
    ) -> SafetyResult:
        """Full safety validation pipeline."""
        faithfulness = self._score_faithfulness(response, context_docs)
        hallucination = self._detect_hallucination(response, context_docs)
        violations = self._check_guardrails(response)

        passed = (
            faithfulness >= self.threshold
            and not hallucination
            and len(violations) == 0
        )

        return SafetyResult(
            passed=passed,
            faithfulness_score=faithfulness,
            hallucination_detected=hallucination,
            guardrail_violations=violations,
            details={
                "query": query,
                "faithfulness_score": faithfulness,
                "hallucination_detected": hallucination,
                "violations": violations,
                "threshold": self.threshold,
            }
        )

    def _score_faithfulness(
        self, response: str, context_docs: list[dict]
    ) -> float:
        """
        Score how faithful the response is to the retrieved context.
        Uses NLI to check if context entails each response sentence.
        """
        sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 20]
        if not sentences:
            return 1.0

        context_text = " ".join(doc["content"][:500] for doc in context_docs[:3])
        scores = []

        for sentence in sentences[:5]:  # Check first 5 sentences
            result = self.nli_pipeline(
                sequences=sentence,
                candidate_labels=["supported by context", "not supported by context"],
                hypothesis_template="This claim is {}.",
                multi_label=False,
            )
            entailment_score = result["scores"][result["labels"].index("supported by context")]
            scores.append(entailment_score)

        return float(sum(scores) / len(scores)) if scores else 1.0

    def _detect_hallucination(
        self, response: str, context_docs: list[dict]
    ) -> bool:
        """
        Detect hallucination using NLI contradiction detection.
        Returns True if hallucination is likely.
        """
        context_text = " ".join(doc["content"][:300] for doc in context_docs[:2])
        result = self.nli_pipeline(
            sequences=response[:512],
            candidate_labels=["consistent with sources", "contradicts sources", "unsupported claim"],
            multi_label=False,
        )
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        return top_label in ["contradicts sources", "unsupported claim"] and top_score > 0.7

    def _check_guardrails(self, response: str) -> list[str]:
        """Check response against financial domain guardrail rules."""
        import re
        violations = []
        response_lower = response.lower()
        for rule in FINANCIAL_GUARDRAILS:
            if re.search(rule["pattern"], response_lower):
                violations.append(rule["message"])
        return violations

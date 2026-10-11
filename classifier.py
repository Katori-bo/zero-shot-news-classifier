"""Zero-shot topic classification with an explicitly selected NLI model."""
from __future__ import annotations

from time import perf_counter
from typing import Any

MODEL_ID = "facebook/bart-large-mnli"
TEMPLATE = "This text is about {}."


def validate_input(text: str, labels: list[str]) -> tuple[str, list[str]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty passage.")
    if not isinstance(labels, (list, tuple)) or not all(
        isinstance(label, str) for label in labels
    ):
        raise ValueError("Candidate labels must be a list of strings.")
    cleaned = [label.strip() for label in labels]
    if len(cleaned) < 2 or any(not label for label in cleaned):
        raise ValueError("Provide at least two non-empty labels.")
    if len({label.casefold() for label in cleaned}) != len(cleaned):
        raise ValueError("Candidate labels must be unique.")
    return text.strip(), cleaned


class NewsClassifier:
    def __init__(self, *, device: int = -1, revision: str | None = None):
        self.device = device
        self.revision = revision
        self._pipeline: Any = None

    def load(self) -> None:
        if self._pipeline is None:
            from transformers import pipeline

            kwargs = {"revision": self.revision} if self.revision else {}
            self._pipeline = pipeline(
                "zero-shot-classification", model=MODEL_ID,
                device=self.device, **kwargs,
            )

    def classify(self, text: str, labels: list[str], *, multi_label=False) -> dict:
        text, labels = validate_input(text, labels)
        self.load()
        tokenizer = self._pipeline.tokenizer
        for label in labels:
            encoded = tokenizer(text, TEMPLATE.format(label))
            if len(encoded["input_ids"]) > tokenizer.model_max_length:
                raise ValueError("The passage and label exceed the model token limit.")
        started = perf_counter()
        result = self._pipeline(
            text, candidate_labels=labels, hypothesis_template=TEMPLATE,
            multi_label=multi_label,
        )
        return {
            "sequence": result["sequence"],
            "labels": result["labels"],
            "scores": [float(score) for score in result["scores"]],
            "model": MODEL_ID,
            "revision": self.revision or getattr(self._pipeline.model.config, "_commit_hash", None),
            "hypothesis_template": TEMPLATE,
            "multi_label": bool(multi_label),
            "inference_seconds": round(perf_counter() - started, 4),
        }

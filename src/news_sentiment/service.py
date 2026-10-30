"""Shared bounded text contract, verified artifacts and transparent weak labels."""

import hashlib
import json
from pathlib import Path

import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LABELS = ["negative", "neutral", "positive"]


def prepare_records(records):
    """Combine headline and optional description consistently for all clients."""
    if not isinstance(records, list) or not 1 <= len(records) <= 200:
        raise ValueError("Provide 1 to 200 records.")
    texts = []
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Row {number}: expected a headline/description object.")
        headline, description = record.get("headline"), record.get("description", "")
        if not isinstance(headline, str) or not headline.strip():
            raise ValueError(f"Row {number}: headline must be a nonempty string.")
        if not isinstance(description, str):
            raise ValueError(f"Row {number}: description must be a string.")
        text = " ".join((headline + " " + description).split())
        if len(text) > 5000 or any(ord(char) < 32 for char in text):
            raise ValueError(
                f"Row {number}: text must be at most 5000 printable characters."
            )
        texts.append(text)
    return texts


def weak_label(compound):
    """Apply VADER's conventional +/-0.05 weak-label boundaries."""
    return (
        "positive"
        if compound >= 0.05
        else "negative" if compound <= -0.05 else "neutral"
    )


class PredictionService:
    """Compare learned label imitation with the rule-based labeling source."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.report = json.loads((model_dir / "metrics.json").read_text())
        path = model_dir / "pipeline.joblib"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != self.report["artifact_sha256"]
        ):
            raise ValueError("Model artifact integrity check failed.")
        self.model = joblib.load(path)
        self.rules = SentimentIntensityAnalyzer()
        self.sample = json.loads((model_dir / "sample.json").read_text())

    def predict(self, records):
        """Return learned scores and VADER labels, without claiming human truth."""
        texts = prepare_records(records)
        predicted = self.model.predict(texts)
        scores = self.model.predict_proba(texts)
        results = []
        for text, label, row in zip(texts, predicted, scores):
            compound = self.rules.polarity_scores(text)["compound"]
            known_terms = self.model.named_steps["tfidfvectorizer"].build_analyzer()(
                text
            )
            vocabulary = self.model.named_steps["tfidfvectorizer"].vocabulary_
            matched = sum(term in vocabulary for term in known_terms)
            results.append(
                {
                    "label": str(label),
                    "scores": {
                        str(key): round(float(value), 6)
                        for key, value in zip(self.model.classes_, row)
                    },
                    "vader_label": weak_label(compound),
                    "vader_compound": compound,
                    "matched_terms": matched,
                    "warning": (
                        "No vocabulary overlap; learned result is unreliable."
                        if matched == 0
                        else None
                    ),
                }
            )
        return results

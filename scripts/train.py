"""Train text models to imitate explicitly disclosed VADER weak labels."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from news_sentiment.service import LABELS, weak_label

ROOT = Path(__file__).resolve().parents[1]


def measure(model, texts, labels):
    """Measure agreement with automated labels, not financial ground truth."""
    predicted = model.predict(texts)
    return {
        "rows": len(labels),
        "accuracy": accuracy_score(labels, predicted),
        "macro_f1": f1_score(labels, predicted, average="macro"),
        "classification_report": classification_report(
            labels, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=LABELS).tolist(),
    }


def train(data_dir, output_dir):
    """Keep duplicate normalized texts together and fit TF-IDF on training only."""
    frames = []
    source_hashes = {}
    for name in [
        "cnbc_headlines.csv",
        "guardian_headlines.csv",
        "reuters_headlines.csv",
    ]:
        path = data_dir / name
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path).fillna("")
        if "Description" not in frame:
            frame["Description"] = ""
        frame["text"] = (
            (frame["Headlines"].astype(str) + " " + frame["Description"].astype(str))
            .str.split()
            .str.join(" ")
        )
        frame["source"] = name
        frames.append(frame[["text", "source"]])
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["text"].str.len().between(1, 5000)].reset_index(drop=True)
    normalized = frame["text"].map(
        lambda text: re.sub(r"\W+", " ", text.lower()).strip()
    )
    groups = pd.factorize(normalized)[0]
    develop, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            frame, groups=groups
        )
    )
    training_rel, validation_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43).split(
            frame.iloc[develop], groups=groups[develop]
        )
    )
    training, validation = develop[training_rel], develop[validation_rel]
    rules = SentimentIntensityAnalyzer()
    labels = frame["text"].map(
        lambda text: weak_label(rules.polarity_scores(text)["compound"])
    )
    partitions = {"train": training, "validation": validation, "test": test}
    for left, right in [(training, validation), (training, test), (validation, test)]:
        if not set(groups[left]).isdisjoint(groups[right]):
            raise ValueError("Normalized text overlap between partitions")
    candidates = {}
    for name, classifier in [
        ("majority_baseline", DummyClassifier(strategy="most_frequent")),
        ("multinomial_naive_bayes", MultinomialNB(alpha=0.5)),
        (
            "logistic_regression",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    ]:
        candidates[name] = make_pipeline(
            TfidfVectorizer(
                max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
            ),
            classifier,
        )
    comparison = {}
    for name, model in candidates.items():
        print("Fitting", name, flush=True)
        model.fit(frame.iloc[training]["text"], labels.iloc[training])
        comparison[name] = measure(
            model, frame.iloc[validation]["text"], labels.iloc[validation]
        )
    winner = max(comparison, key=lambda key: comparison[key]["macro_f1"])
    model = candidates[winner]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pipeline.joblib"
    joblib.dump(model, path, compress=3)
    report = {
        "version": "1.0.0",
        "selected_model": winner,
        "classes": LABELS,
        "label_source": (
            "vaderSentiment 3.3.2, compound boundaries +/-0.05; "
            "automated weak labels"
        ),
        "source_sha256": source_hashes,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
        "split_strategy": (
            "grouped normalized text 60/20/20, seed 42/43; " "not temporal/event-level"
        ),
        "split": {
            name: {
                "rows": len(idx),
                "class_counts": labels.iloc[idx].value_counts().to_dict(),
            }
            for name, idx in partitions.items()
        },
        "validation_candidates": comparison,
        "test": measure(model, frame.iloc[test]["text"], labels.iloc[test]),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "sample.json").write_text(
        json.dumps(
            {
                "headline": "Company reports higher earnings after strong sales",
                "description": (
                    "A synthetic example for learning, " "not a real news report."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report["test"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)

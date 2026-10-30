"""Text contract, weak-label transparency and HTTP regression checks."""

import io
from pathlib import Path
import pytest
from news_sentiment.service import PredictionService, prepare_records, weak_label
from news_sentiment.web import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(name="service", scope="module")
def fixture_service():
    """Load the reviewed pipeline."""
    return PredictionService(ROOT / "models")


@pytest.fixture(name="client")
def fixture_client():
    """Create isolated HTTP client."""
    return create_app({"TESTING": True, "MODEL_DIR": ROOT / "models"}).test_client()


@pytest.mark.parametrize(
    "records",
    [
        None,
        [],
        {},
        [None],
        [{}],
        [{"headline": ""}],
        [{"headline": 42}],
        [{"headline": "x", "description": None}],
        [{"headline": "x" * 5001}],
        [{"headline": "x"}] * 201,
    ],
)
def test_invalid_records(records):
    """Reject malformed, empty and excessive text batches."""
    with pytest.raises(ValueError):
        prepare_records(records)


@pytest.mark.parametrize(
    "compound,label",
    [
        (-1, "negative"),
        (-0.05, "negative"),
        (0, "neutral"),
        (0.049, "neutral"),
        (0.05, "positive"),
        (1, "positive"),
    ],
)
def test_weak_label_boundaries(compound, label):
    """Weak supervision uses the disclosed exact boundary convention."""
    assert weak_label(compound) == label


def test_combining_and_normalization():
    """Use one combined input definition across training and serving."""
    assert prepare_records(
        [{"headline": "  Earnings rise ", "description": " Sales grow. "}]
    ) == ["Earnings rise Sales grow."]


def test_inference_and_batch_parity(service):
    """Retain both the learned label and its rule-based reference."""
    result = service.predict([service.sample])[0]
    assert service.predict([service.sample] * 3) == [result] * 3
    assert result["label"] in ["negative", "neutral", "positive"]
    assert abs(sum(result["scores"].values()) - 1) < 0.00001
    assert "vader_label" in result and "vader_compound" in result


def test_unknown_vocabulary_warning(service):
    """Do not present intercept-only predictions without a warning."""
    result = service.predict([{"headline": "zzzzqwxxyyzz"}])[0]
    assert result["matched_terms"] == 0
    assert result["warning"]


def test_html_api_and_csv(client, service):
    """All three interfaces use the same reviewed service."""
    home = client.get("/")
    assert home.status_code == 200 and b"Financial News Lens" in home.data
    response = client.post("/predict", data=service.sample)
    assert response.status_code == 200 and b"LEARNED MODEL" in response.data
    response = client.post("/api/predict", json={"records": [service.sample]})
    assert response.json["predictions"] == service.predict([service.sample])
    assert "not human" in response.json["evaluation_target"]
    sample = client.get("/sample.csv")
    response = client.post(
        "/predict-csv", data={"file": (io.BytesIO(sample.data), "sample.csv")}
    )
    assert response.status_code == 200 and b"vader_label" in response.data
    assert service.sample["headline"].encode() not in response.data


def test_errors_limits_and_escape(client):
    """Error handling, escaping and input limits do not leak raw text."""
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.post("/predict", data={"headline": ""}).status_code == 400
    assert client.post("/predict-csv").status_code == 400
    assert client.get("/missing").status_code == 404
    assert (
        client.post(
            "/api/predict",
            data=b"x" * (1024 * 1024 + 1),
            content_type="application/json",
        ).status_code
        == 413
    )
    html = client.post("/predict", data={"headline": "<script>alert(1)</script>"}).data
    assert b"<script>alert(1)</script>" not in html


def test_metrics_health_and_headers(client):
    """Published results describe weak-label agreement and match row totals."""
    report = client.get("/api/metrics").json
    assert "automated weak" in report["label_source"]
    assert sum(map(sum, report["test"]["confusion_matrix"])) == report["test"]["rows"]
    response = client.get("/health/ready")
    assert response.json["status"] == "ready"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_model_integrity(tmp_path):
    """Reject a mismatched artifact before deserialization."""
    (tmp_path / "metrics.json").write_bytes((ROOT / "models/metrics.json").read_bytes())
    (tmp_path / "pipeline.joblib").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        PredictionService(tmp_path)

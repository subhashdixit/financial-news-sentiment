"""Flask routes for transparent financial-news weak-label exploration."""

import logging
import os
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from news_sentiment.service import PredictionService


def create_app(config=None):
    """Load one trusted model and configure bounded text input."""
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=1024 * 1024,
        MODEL_DIR=os.environ.get("MODEL_DIR", str(Path.cwd() / "models")),
    )
    app.config.update(config or {})
    logging.basicConfig(level=logging.INFO)
    service = PredictionService(app.config["MODEL_DIR"])
    app.extensions["prediction_service"] = service

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; "
            "frame-ancestors 'none'; form-action 'self'"
        )
        return response

    @app.get("/")
    def home():
        return render_template(
            "index.html",
            values=service.sample,
            result=None,
            error=None,
            report=service.report,
        )

    @app.post("/predict")
    def form():
        values = {key: request.form.get(key, "") for key in ["headline", "description"]}
        try:
            result = service.predict([values])[0]
        except ValueError as exc:
            return (
                render_template(
                    "index.html",
                    values=values,
                    result=None,
                    error=str(exc),
                    report=service.report,
                ),
                400,
            )
        return render_template(
            "index.html",
            values=values,
            result=result,
            error=None,
            report=service.report,
        )

    @app.post("/api/predict")
    def predict_json():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "records" not in payload:
            raise ValueError("Expected JSON object containing records list.")
        return jsonify(
            predictions=service.predict(payload["records"]),
            version=service.report["version"],
            evaluation_target="automated VADER labels, not human financial sentiment",
        )

    @app.get("/sample.csv")
    def sample():
        return Response(
            pd.DataFrame([service.sample]).to_csv(index=False),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=sample.csv"},
        )

    @app.post("/predict-csv")
    def csv_score():
        uploaded = request.files.get("file")
        if uploaded is None:
            raise ValueError("Upload a CSV in the file field.")
        try:
            frame = pd.read_csv(uploaded, nrows=201, keep_default_na=False)
        except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
            raise ValueError(
                "Cannot read CSV; use UTF-8 and a headline column."
            ) from exc
        results = service.predict(frame.to_dict(orient="records"))
        rows = [
            {
                "row": idx,
                "label": item["label"],
                "vader_label": item["vader_label"],
                "vader_compound": item["vader_compound"],
                **item["scores"],
            }
            for idx, item in enumerate(results, 1)
        ]
        return Response(
            pd.DataFrame(rows).to_csv(index=False),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=sentiment.csv"},
        )

    @app.get("/api/metrics")
    def metrics():
        return jsonify(service.report)

    @app.get("/health/ready")
    def ready():
        return jsonify(
            status="ready",
            version=service.report["version"],
            revision=os.environ.get("RENDER_GIT_COMMIT", "local"),
        )

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(Exception)
    def unexpected(error):
        app.logger.error("Unhandled error", exc_info=error)
        return jsonify(error="Unexpected server error."), 500

    return app

"""Gunicorn and local Flask entry point."""

from news_sentiment.web import create_app

app = create_app()

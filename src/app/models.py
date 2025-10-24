# src/app/models.py
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class SentimentEmbedModel:
    def __init__(self):
        model_name = os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment = pipeline("sentiment-analysis", model=model_name, device=-1)
        embed_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(embed_model)

    def analyze_sentiment(self, text):
        result = self.sentiment(text, truncation=True)
        if result and isinstance(result, list):
            return result[0]
        return {"label": "NEUTRAL", "score": 0.0}

    def embed_text(self, text):
        return self.embedder.encode(text, convert_to_numpy=True)

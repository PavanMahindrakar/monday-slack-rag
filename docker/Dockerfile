# docker/Dockerfile

FROM python:3.10-slim

# -------------------------
# 1️⃣ Set working directory
# -------------------------
WORKDIR /app

# -------------------------
# 2️⃣ Install system dependencies (needed for torch, transformers, FAISS)
# -------------------------
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# 3️⃣ Copy and install Python dependencies
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# 4️⃣ Pre-download both Hugging Face models for caching
# -------------------------
# DistilBERT (fast fallback)
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', revision='714eb0f')"

# CardiffNLP (preferred advanced model)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
tokenizer=AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment'); \
model=AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')"

# -------------------------
# 5️⃣ Copy source code
# -------------------------
COPY src/ /app

# -------------------------
# 6️⃣ Default CMD for API container
# -------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

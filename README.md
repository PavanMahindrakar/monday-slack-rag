# 🚀 Monday–Slack AI Sentiment System

> AI-powered hybrid automation for real-time task sentiment analysis, Slack alerts, and semantic search using RAG (Retrieval-Augmented Generation).

![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![FastAPI](https://img.shields.io/badge/FastAPI-Enabled-009688?logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)
![RQ](https://img.shields.io/badge/RQ-Worker-red)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-green)

---

## 🧭 Overview

**Monday–Slack AI Sentiment System** is an intelligent monitoring pipeline that connects to **Monday.com** boards, detects **positive, neutral, or negative sentiment** in tasks, sends **real-time Slack alerts**, stores vector embeddings in **FAISS**, and enables **semantic search** using RAG (Retrieval-Augmented Generation).
It combines **domain-specific keyword intelligence** with **Transformer-based models** for robust hybrid sentiment detection.

---

## 🧠 Architecture

### 🗺️ System Flow Diagram

![System Architecture](A_flowchart_diagram_titled_"Monday–Slack_AI_Sentim.png")

```
Monday.com → FastAPI Webhook → Redis Queue → Worker (NLP & FAISS)
                                   ↓
                            Slack Notifications
                                   ↓
                         Semantic Search (RAG Engine)
```

---

## ⚙️ Features

| Feature                          | Description                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------- |
| 🧩 **Hybrid Sentiment Analysis** | Combines domain-specific keyword logic + Transformer models (CardiffNLP / DistilBERT) |
| 💬 **Slack Alerts**              | Real-time alerts for highly positive or negative feedback                             |
| 💾 **FAISS Vector Store**        | Stores embeddings for every analyzed task                                             |
| 🔍 **Semantic Search (RAG)**     | Enables retrieval & summarization of historical tasks                                 |
| ⏰ **RQ Scheduler**               | Weekly sentiment summary every Monday 9 AM IST                                        |
| 🐳 **Dockerized Setup**          | One-command containerized deployment with Docker Compose                              |

---

## 🧰 Tech Stack

* **Python 3.10+**
* **FastAPI** – REST API for Monday webhook and semantic search
* **Redis + RQ** – Background task queue and scheduler
* **HuggingFace Transformers** – For sentiment analysis (CardiffNLP / DistilBERT)
* **SentenceTransformers (MiniLM)** – For generating embeddings
* **FAISS** – High-speed similarity search engine
* **Slack Webhooks** – For alert notifications
* **Docker Compose** – To orchestrate multi-service setup

---

## 📦 Folder Structure

```
MONDAY-SLACK-RAG/
├── docker-compose.yml
├── docker/
│   └── Dockerfile
├── src/
│   └── app/
│       ├── main.py          # FastAPI server + RAG API
│       ├── tasks.py         # Worker logic (NLP, Slack, FAISS)
│       ├── storage.py       # FAISS vector DB wrapper
│       ├── alerts.py        # Slack alert helper
│       ├── rag.py           # Semantic search / RAG engine
│       ├── models.py, utils.py, auto_keywords.py
├── data/                    # Vector index storage (ignored in git)
├── cache/
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/monday-slack-rag.git
cd monday-slack-rag
```

### 2️⃣ Create `.env` File

```env
MONDAY_API_TOKEN=your_token_here
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
REDIS_URL=redis://redis:6379/0
VECTOR_DB_PATH=/data/faiss_index
PORT=8080
```

> ⚠️ Do not push your `.env` file to GitHub — it contains credentials.

### 3️⃣ Build and Run Docker Containers

```bash
docker-compose up --build
```

Once running, FastAPI will start at:
👉 **[http://localhost:8080](http://localhost:8080)**

---

## 🌍 Expose to Monday Webhook (Ngrok)

You can expose your FastAPI app to Monday.com using **ngrok**:

```bash
ngrok http 8080
```

Then copy your ngrok HTTPS URL and configure it in Monday.com’s **Webhook Integration** (pointing to `/webhook/monday`).

---

## 🔍 Semantic Search (RAG Mode)

After data is stored, you can query semantically similar tasks using:

**Endpoint:**

```
GET /search?query=<your_query>
```

**Example:**

```bash
curl "http://localhost:8080/search?query=bad%20performance"
```

**Output:**

```json
{
  "query": "bad performance",
  "results": [
    {"text": "app keeps crashing after update", "sentiment": {"label": "NEGATIVE", "score": 0.99}},
    {"text": "slow response when logging in", "sentiment": {"label": "NEGATIVE", "score": 0.96}}
  ]
}
```

---

## 🕒 Weekly Summary Report

Every **Monday at 9 AM IST**, the system automatically sends a summary to Slack:

```
📊 Weekly Sentiment Summary
Total Items Analyzed: 45
Positive: 30
Negative: 10
Neutral: 5
```

---

## 📈 Logs & Monitoring

| Command                                       | Description                        |
| --------------------------------------------- | ---------------------------------- |
| `docker logs -f ms_api`                       | Live FastAPI logs                  |
| `docker logs -f ms_worker`                    | Monitor NLP + sentiment processing |
| `docker exec -it ms_worker rq info`           | Check Redis queue status           |
| `docker exec -it ms_worker rq-scheduler info` | Verify scheduled jobs              |
| `docker exec -it ms_worker ls /data`          | Verify FAISS index file            |

---

## 💡 RAG Enhancement Overview

RAG (Retrieval-Augmented Generation) allows semantic querying across all stored feedback.

**Flow:**

1. Encode query → embedding using MiniLM
2. Retrieve similar vectors via FAISS
3. (Optional) Summarize using LLM

This enables:

* Searching for “login issues” or “crash reports”
* Generating summaries like “Top 5 negative feedback themes”

---

## 🪄 Future Enhancements

* 🧠 Integrate Local LLM (Mistral / Ollama) for offline reasoning
* 🧩 Add LangChain retriever pipeline for structured RAG
* 📊 Create React + Tailwind dashboard for live analytics
* 🧾 Export Slack summaries to Google Sheets
* 🔍 Add clustering and trend detection

---

## 🤝 Contributing

Pull requests are welcome!
For major changes, open an issue to discuss improvements before submitting a PR.

---

## 📜 License

MIT License © 2025 — Developed by **Pavan Mahindrakar**

---

> 🧩 *Turning team feedback into insight — automatically.*

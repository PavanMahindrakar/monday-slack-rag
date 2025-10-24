# src/app/rag.py
import os, requests
from fastapi import FastAPI
from models import SentimentEmbedModel
from storage import VectorStore

app = FastAPI()
sem = SentimentEmbedModel()
vs = VectorStore()

@app.post("/rag/explain")
async def explain(payload: dict):
    qtext = payload.get("text")
    if not qtext:
        return {"error": "text required"}
    qvec = sem.embed_text(qtext)
    results = vs.search(qvec, k=5)
    context = "\n\n".join([r.get("text","") for r in results])
    prompt = f"Context:\n{context}\n\nQuestion: {qtext}\nAnswer concisely referencing the context."
    # If you have a local LLM API set LLM_URL env; otherwise return context + simple summary
    LLM_URL = os.getenv("LLM_URL")
    if LLM_URL:
        resp = requests.post(LLM_URL, json={"prompt": prompt}, timeout=30)
        return resp.json()
    else:
        # fallback: return context plus brief heuristic
        return {"context": context, "hint": "No LLM configured - set LLM_URL to call your model."}

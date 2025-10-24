# # # # # src/app/tasks.py
# # # # src/app/tasks.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import re
import json
import requests
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
from redis import Redis
from rq import Queue
from rq_scheduler import Scheduler
from sentence_transformers import SentenceTransformer
from app.storage import VectorStore

load_dotenv()

# ------------- Redis / Queue -------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
q = Queue("monday", connection=redis_conn)
scheduler = Scheduler(queue=q, connection=redis_conn)

# ------------- Config -------------
MONDAY_API_URL = "https://api.monday.com/v2"
MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "").strip()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
ALERT_NEGATIVE_THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))

# ------------- Sentiment Models -------------
USE_CARDIFFNLP = False
USE_DISTILBERT = False
hf_pipeline = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    hf_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    USE_CARDIFFNLP = True
    print("✅ Using CardiffNLP/twitter-roberta-base-sentiment model.")
except Exception as e1:
    print("⚠️ CardiffNLP unavailable:", e1)
    try:
        from transformers import pipeline
        hf_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        )
        USE_DISTILBERT = True
        print("✅ Using DistilBERT SST-2 fallback model.")
    except Exception as e2:
        hf_pipeline = None
        print("❌ No transformer model available:", e2)

# ✅ Sentence embedding model
try:
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Sentence embedding model loaded (all-MiniLM-L6-v2).")
except Exception as e:
    embedder = None
    print("⚠️ Could not load SentenceTransformer:", e)

# ------------- Keyword Patterns -------------
NEGATIVE_KEYWORDS = [
    "bad", "awful", "terrible", "worst", "fail", "failure", "wrong", "problem",
    "error", "issue", "delay", "hate", "not good", "not working", "stuck",
    "broken", "poor", "disappointed", "unhappy", "hard", "bug", "slow",
    "crash", "horrible", "sucks", "waste", "not satisfied", "useless"
]
POSITIVE_KEYWORDS = [
    "good", "great", "awesome", "amazing", "excellent", "love", "fantastic",
    "perfect", "well done", "nice", "superb", "wonderful", "brilliant", "cool",
    "outstanding", "happy", "impressive"
]

# Compile regex
def compile_patterns():
    global _negative_patterns, _positive_patterns
    _negative_patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in NEGATIVE_KEYWORDS]
    _positive_patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in POSITIVE_KEYWORDS]

compile_patterns()

# ------------- Auto Keyword Learning -------------
def learn_keywords(min_freq=2):
    try:
        vs = VectorStore()
        all_meta = vs.meta
        pos_words, neg_words = [], []

        def clean_text(t):
            return re.sub(r"[^a-zA-Z0-9\s]", "", t.lower())

        for m in all_meta:
            label = m["sentiment"]["label"].upper()
            words = clean_text(m["text"]).split()
            if label == "POSITIVE":
                pos_words.extend(words)
            elif label == "NEGATIVE":
                neg_words.extend(words)

        pos_common = [w for w, c in Counter(pos_words).items() if c >= min_freq and len(w) > 3]
        neg_common = [w for w, c in Counter(neg_words).items() if c >= min_freq and len(w) > 3]

        pos_common = [w for w in pos_common if w not in NEGATIVE_KEYWORDS]
        neg_common = [w for w in neg_common if w not in POSITIVE_KEYWORDS]

        print(f"🧠 Auto-learned keywords → +{len(pos_common)} positive, +{len(neg_common)} negative")

        # Merge into keyword lists
        POSITIVE_KEYWORDS.extend([w for w in pos_common if w not in POSITIVE_KEYWORDS])
        NEGATIVE_KEYWORDS.extend([w for w in neg_common if w not in NEGATIVE_KEYWORDS])

        compile_patterns()
        return {"positive": pos_common, "negative": neg_common}
    except Exception as e:
        print("⚠️ Auto keyword learning failed:", e)
        return {"positive": [], "negative": []}

# ------------- Hybrid Sentiment Detection -------------
def predict_sentiment_combined(text):
    text_str = (text or "").strip()
    if not text_str:
        return {"label": "NEUTRAL", "score": 0.0}

    # 🧠 Keyword override
    neg_match = any(p.search(text_str) for p in _negative_patterns)
    pos_match = any(p.search(text_str) for p in _positive_patterns)
    if neg_match and not pos_match:
        print(f"🧠 Keyword override: NEGATIVE → '{text_str}'")
        return {"label": "NEGATIVE", "score": 0.995}
    elif pos_match and not neg_match:
        print(f"🧠 Keyword override: POSITIVE → '{text_str}'")
        return {"label": "POSITIVE", "score": 0.995}

    # 🤖 Transformer fallback
    if hf_pipeline is None:
        print("⚠️ No model loaded — defaulting to NEUTRAL.")
        return {"label": "NEUTRAL", "score": 0.0}

    try:
        out = hf_pipeline(text_str, truncation=True)
        result = out[0]
        label_raw = str(result.get("label", "")).upper()
        score = float(result.get("score", 0.0))

        if "LABEL" in label_raw:
            if label_raw.endswith("0"):
                label = "NEGATIVE"
            elif label_raw.endswith("1"):
                label = "NEUTRAL"
            elif label_raw.endswith("2"):
                label = "POSITIVE"
            else:
                label = "NEUTRAL"
        else:
            label = label_raw if label_raw in ("NEGATIVE", "POSITIVE", "NEUTRAL") else "NEUTRAL"

        print(f"🤖 Model ({'Cardiff' if USE_CARDIFFNLP else 'DistilBERT'}) output: {label_raw} → {label} ({score:.2f})")

        if score < 0.55:
            return {"label": "NEUTRAL", "score": score}

        return {"label": label, "score": score}
    except Exception as e:
        print("⚠️ Sentiment model error:", e)
        return {"label": "NEUTRAL", "score": 0.0}

# ------------- Monday API Helper -------------
def fetch_monday_item_details(board_id, item_id):
    if not MONDAY_API_TOKEN:
        print("⚠️ MONDAY_API_TOKEN not set; cannot fetch item details.")
        return None

    headers = {
        "Authorization": MONDAY_API_TOKEN,
        "Content-Type": "application/json"
    }

    query = """
    query ($board_id: [ID!], $item_id: [ID!]) {
      boards (ids: $board_id) {
        name
        items_page (limit: 1, query_params: {ids: $item_id}) {
          items {
            name
            creator {
              name
              email
            }
          }
        }
      }
    }
    """

    variables = {"board_id": [str(board_id)], "item_id": [str(item_id)]}

    try:
        resp = requests.post(
            MONDAY_API_URL, headers=headers,
            json={"query": query, "variables": variables}, timeout=10
        )
        data = resp.json()
        if "errors" in data:
            print("⚠️ Monday API returned errors:", data["errors"])
            return None

        board = data["data"]["boards"][0]
        items = board["items_page"]["items"]
        if not items:
            return {"board_name": board.get("name", "N/A")}
        item = items[0]
        creator = item.get("creator") or {}
        return {
            "board_name": board.get("name", "N/A"),
            "item_name": item.get("name", "N/A"),
            "creator_name": creator.get("name", "Unknown"),
            "creator_email": creator.get("email", "-"),
        }
    except Exception as e:
        print("⚠️ Failed to fetch item details:", e)
        try:
            print("🧾 [DEBUG] Monday API raw response:", resp.text)
        except:
            pass
        return None

# ------------- Queue Function -------------
def enqueue_monday_event(event: dict):
    print("📩 [API] Received monday event, queueing job...")
    try:
        q.enqueue("app.tasks.process_event", event)
    except Exception:
        q.enqueue(process_event, event)

# ------------- VectorStore Helper -------------
def add_vector_to_store(vec, meta):
    try:
        vs = VectorStore()
        vs.add_vector(vec, meta)
        print("💾 [VectorStore] Added vector successfully.")
    except Exception as e:
        print("⚠️ [VectorStore] Failed to add vector:", e)

# ------------- Worker Logic -------------
def process_event(event: dict):
    print("🟢 [Worker] Processing event:", event)
    try:
        evt = event.get("event", {})
        board_id = evt.get("boardId")
        item_id = evt.get("pulseId") or evt.get("itemId")

        details = fetch_monday_item_details(board_id, item_id)
        if details and details.get("item_name") and details["item_name"].lower() != "new task":
            text = details["item_name"]
            print(f"🧾 [DEBUG] Using Monday API-fetched text: '{text}'")
        else:
            text = evt.get("pulseName") or evt.get("itemName") or evt.get("text") or json.dumps(evt)[:2000]
            print(f"🧾 [DEBUG] Extracted text for sentiment: '{text}'")

        sentiment = predict_sentiment_combined(text)
        label = sentiment.get("label", "NEUTRAL").upper()
        score = float(sentiment.get("score", 0.0))
        print(f"🔍 Final Sentiment: {label} ({score})")

        if embedder:
            embedding = embedder.encode(text).tolist()
        else:
            embedding = [score]
        add_vector_to_store(embedding, {"text": text, "sentiment": sentiment})

        board_name = details.get("board_name") if details else "N/A"
        item_name = details.get("item_name") if details else text[:80]
        creator_name = details.get("creator_name") if details else "Unknown"
        creator_email = details.get("creator_email") if details else "-"
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        slack_text = (
            f"*{'🚨 Negative' if label == 'NEGATIVE' else '🎉 Positive'} Sentiment Detected*\n"
            f"*Board:* {board_name}\n"
            f"*Item:* {item_name}\n"
            f"*Created by:* {creator_name} ({creator_email})\n"
            f"*Sentiment:* `{label}` ({score:.2f})\n"
            f"*Time:* {timestamp}\n\n"
            "_Generated automatically by your Monday–Slack AI Monitor 🤖_"
        )

        if SLACK_WEBHOOK_URL:
            if label == "NEGATIVE" and score >= ALERT_NEGATIVE_THRESHOLD:
                requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
                print("🚨 Slack alert sent for NEGATIVE sentiment.")
            elif label == "POSITIVE":
                requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
                print("🎉 Slack alert sent for POSITIVE sentiment.")
        else:
            print("⚠️ SLACK_WEBHOOK_URL not configured.")

        print("💾 Processing complete.")
    except Exception as e:
        print("❌ Error processing event:", e)

# ------------- Scheduler (Weekly Summary) -------------
def send_weekly_summary():
    vs = VectorStore()
    total = len(vs.meta)
    neg = sum(1 for m in vs.meta if m["sentiment"]["label"] == "NEGATIVE")
    pos = sum(1 for m in vs.meta if m["sentiment"]["label"] == "POSITIVE")

    summary = f"""
📊 *Weekly Sentiment Summary*
Total Items Analyzed: {total}
Positive: {pos}
Negative: {neg}
Neutral: {total - pos - neg}

_Week ending: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}_"""

    # 🔁 Auto-learn new keywords weekly
    learned = learn_keywords(min_freq=2)
    if learned["positive"] or learned["negative"]:
        summary += "\n\n🧠 *Auto-Learned Keywords:*\n"
        if learned["positive"]:
            summary += f"Positive ➕: {', '.join(learned['positive'][:10])}\n"
        if learned["negative"]:
            summary += f"Negative ➖: {', '.join(learned['negative'][:10])}\n"

    if SLACK_WEBHOOK_URL:
        requests.post(SLACK_WEBHOOK_URL, json={"text": summary})
        print("📤 Weekly summary sent to Slack.")
    else:
        print(summary)

try:
    for job in scheduler.get_jobs():
        if job.id == "weekly_summary_job":
            scheduler.cancel(job)
    scheduler.cron(
        "0 9 * * MON",
        func=send_weekly_summary,
        queue_name="monday",
        id="weekly_summary_job",
    )
    print("🕒 Weekly summary job scheduled (Mondays 9 AM IST).")
except Exception as e:
    print("⚠️ Could not schedule weekly summary:", e)





# Perfect one ----------------------------------------------------------------------
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# import os
# import re
# import json
# import requests
# from datetime import datetime
# from dotenv import load_dotenv
# from redis import Redis
# from rq import Queue
# from rq_scheduler import Scheduler
# from sentence_transformers import SentenceTransformer
# from app.storage import VectorStore

# load_dotenv()

# # ------------- Redis / Queue -------------
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# redis_conn = Redis.from_url(REDIS_URL)
# q = Queue("monday", connection=redis_conn)
# scheduler = Scheduler(queue=q, connection=redis_conn)

# # ------------- Config -------------
# MONDAY_API_URL = "https://api.monday.com/v2"
# MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "").strip()
# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
# ALERT_NEGATIVE_THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))

# # ------------- Sentiment Models -------------
# USE_CARDIFFNLP = False
# USE_DISTILBERT = False
# hf_pipeline = None

# try:
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
#     model_name = "cardiffnlp/twitter-roberta-base-sentiment"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     hf_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
#     USE_CARDIFFNLP = True
#     print("✅ Using CardiffNLP/twitter-roberta-base-sentiment model.")
# except Exception as e1:
#     print("⚠️ CardiffNLP unavailable:", e1)
#     try:
#         from transformers import pipeline
#         hf_pipeline = pipeline(
#             "sentiment-analysis",
#             model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#         )
#         USE_DISTILBERT = True
#         print("✅ Using DistilBERT SST-2 fallback model.")
#     except Exception as e2:
#         hf_pipeline = None
#         print("❌ No transformer model available:", e2)

# # ✅ Sentence embedding model
# try:
#     embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     print("✅ Sentence embedding model loaded (all-MiniLM-L6-v2).")
# except Exception as e:
#     embedder = None
#     print("⚠️ Could not load SentenceTransformer:", e)

# # ------------- Keyword Patterns -------------
# NEGATIVE_KEYWORDS = [
#     "bad", "awful", "terrible", "worst", "fail", "failure", "wrong", "problem",
#     "error", "issue", "delay", "hate", "not good", "not working", "stuck",
#     "broken", "poor", "disappointed", "unhappy", "hard", "bug", "slow",
#     "crash", "horrible", "sucks", "waste", "not satisfied", "useless"
# ]
# POSITIVE_KEYWORDS = [
#     "good", "great", "awesome", "amazing", "excellent", "love", "fantastic",
#     "perfect", "well done", "nice", "superb", "wonderful", "brilliant", "cool",
#     "outstanding", "happy", "impressive"
# ]

# _negative_patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in NEGATIVE_KEYWORDS]
# _positive_patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in POSITIVE_KEYWORDS]

# # ------------- Hybrid Sentiment Detection -------------
# def predict_sentiment_combined(text):
#     text_str = (text or "").strip()
#     if not text_str:
#         return {"label": "NEUTRAL", "score": 0.0}

#     # 🧠 Keyword override
#     neg_match = any(p.search(text_str) for p in _negative_patterns)
#     pos_match = any(p.search(text_str) for p in _positive_patterns)
#     if neg_match and not pos_match:
#         print(f"🧠 Keyword override: NEGATIVE → '{text_str}'")
#         return {"label": "NEGATIVE", "score": 0.995}
#     elif pos_match and not neg_match:
#         print(f"🧠 Keyword override: POSITIVE → '{text_str}'")
#         return {"label": "POSITIVE", "score": 0.995}

#     # 🤖 Transformer fallback
#     if hf_pipeline is None:
#         print("⚠️ No model loaded — defaulting to NEUTRAL.")
#         return {"label": "NEUTRAL", "score": 0.0}

#     try:
#         out = hf_pipeline(text_str, truncation=True)
#         result = out[0]
#         label_raw = str(result.get("label", "")).upper()
#         score = float(result.get("score", 0.0))

#         if "LABEL" in label_raw:
#             if label_raw.endswith("0"):
#                 label = "NEGATIVE"
#             elif label_raw.endswith("1"):
#                 label = "NEUTRAL"
#             elif label_raw.endswith("2"):
#                 label = "POSITIVE"
#             else:
#                 label = "NEUTRAL"
#         else:
#             label = label_raw if label_raw in ("NEGATIVE", "POSITIVE", "NEUTRAL") else "NEUTRAL"

#         print(f"🤖 Model ({'Cardiff' if USE_CARDIFFNLP else 'DistilBERT'}) output: {label_raw} → {label} ({score:.2f})")

#         if score < 0.55:
#             return {"label": "NEUTRAL", "score": score}

#         return {"label": label, "score": score}

#     except Exception as e:
#         print("⚠️ Sentiment model error:", e)
#         return {"label": "NEUTRAL", "score": 0.0}

# # ------------- Monday API Helper (✅ Updated for new GraphQL schema) -------------
# def fetch_monday_item_details(board_id, item_id):
#     if not MONDAY_API_TOKEN:
#         print("⚠️ MONDAY_API_TOKEN not set; cannot fetch item details.")
#         return None

#     headers = {
#         "Authorization": MONDAY_API_TOKEN,
#         "Content-Type": "application/json"
#     }

#     query = """
#     query ($board_id: [ID!], $item_id: [ID!]) {
#       boards (ids: $board_id) {
#         name
#         items_page (limit: 1, query_params: {ids: $item_id}) {
#           items {
#             name
#             creator {
#               name
#               email
#             }
#           }
#         }
#       }
#     }
#     """

#     variables = {
#         "board_id": [str(board_id)],
#         "item_id": [str(item_id)]
#     }

#     try:
#         resp = requests.post(
#             MONDAY_API_URL,
#             headers=headers,
#             json={"query": query, "variables": variables},
#             timeout=10
#         )
#         data = resp.json()

#         if "errors" in data:
#             print("⚠️ Monday API returned errors:", data["errors"])
#             return None

#         board = data["data"]["boards"][0]
#         items = board["items_page"]["items"]
#         if not items:
#             return {"board_name": board.get("name", "N/A")}

#         item = items[0]
#         creator = item.get("creator") or {}

#         return {
#             "board_name": board.get("name", "N/A"),
#             "item_name": item.get("name", "N/A"),
#             "creator_name": creator.get("name", "Unknown"),
#             "creator_email": creator.get("email", "-"),
#         }

#     except Exception as e:
#         print("⚠️ Failed to fetch item details:", e)
#         try:
#             print("🧾 [DEBUG] Monday API raw response:", resp.text)
#         except:
#             pass
#         return None

# # ------------- Queue Function -------------
# def enqueue_monday_event(event: dict):
#     print("📩 [API] Received monday event, queueing job...")
#     try:
#         q.enqueue("app.tasks.process_event", event)
#     except Exception:
#         q.enqueue(process_event, event)

# # ------------- VectorStore Helper -------------
# def add_vector_to_store(vec, meta):
#     try:
#         vs = VectorStore()
#         vs.add_vector(vec, meta)
#         print("💾 [VectorStore] Added vector successfully.")
#     except Exception as e:
#         print("⚠️ [VectorStore] Failed to add vector:", e)

# # ------------- Worker Logic -------------
# def process_event(event: dict):
#     print("🟢 [Worker] Processing event:", event)
#     try:
#         evt = event.get("event", {})
#         board_id = evt.get("boardId")
#         item_id = evt.get("pulseId") or evt.get("itemId")

#         details = fetch_monday_item_details(board_id, item_id)
#         if details and details.get("item_name") and details["item_name"].lower() != "new task":
#             text = details["item_name"]
#             print(f"🧾 [DEBUG] Using Monday API-fetched text: '{text}'")
#         else:
#             text = evt.get("pulseName") or evt.get("itemName") or evt.get("text") or json.dumps(evt)[:2000]
#             print(f"🧾 [DEBUG] Extracted text for sentiment: '{text}'")

#         sentiment = predict_sentiment_combined(text)
#         label = sentiment.get("label", "NEUTRAL").upper()
#         score = float(sentiment.get("score", 0.0))
#         print(f"🔍 Final Sentiment: {label} ({score})")

#         if embedder:
#             embedding = embedder.encode(text).tolist()
#         else:
#             embedding = [score]
#         add_vector_to_store(embedding, {"text": text, "sentiment": sentiment})

#         board_name = details.get("board_name") if details else "N/A"
#         item_name = details.get("item_name") if details else text[:80]
#         creator_name = details.get("creator_name") if details else "Unknown"
#         creator_email = details.get("creator_email") if details else "-"
#         timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

#         slack_text = (
#             f"*{'🚨 Negative' if label == 'NEGATIVE' else '🎉 Positive'} Sentiment Detected*\n"
#             f"*Board:* {board_name}\n"
#             f"*Item:* {item_name}\n"
#             f"*Created by:* {creator_name} ({creator_email})\n"
#             f"*Sentiment:* `{label}` ({score:.2f})\n"
#             f"*Time:* {timestamp}\n\n"
#             "_Generated automatically by your Monday–Slack AI Monitor 🤖_"
#         )

#         if SLACK_WEBHOOK_URL:
#             if label == "NEGATIVE" and score >= ALERT_NEGATIVE_THRESHOLD:
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🚨 Slack alert sent for NEGATIVE sentiment.")
#             elif label == "POSITIVE":
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🎉 Slack alert sent for POSITIVE sentiment.")
#         else:
#             print("⚠️ SLACK_WEBHOOK_URL not configured.")

#         print("💾 Processing complete.")
#     except Exception as e:
#         print("❌ Error processing event:", e)

# # ------------- Scheduler (Weekly Summary) -------------
# from app.storage import VectorStore

# def send_weekly_summary():
#     vs = VectorStore()
#     total = len(vs.meta)
#     neg = sum(1 for m in vs.meta if m["sentiment"]["label"] == "NEGATIVE")
#     pos = sum(1 for m in vs.meta if m["sentiment"]["label"] == "POSITIVE")

#     summary = f"""
# 📊 *Weekly Sentiment Summary*
# Total Items Analyzed: {total}
# Positive: {pos}
# Negative: {neg}
# Neutral: {total - pos - neg}

# _Week ending: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}_"""

#     if SLACK_WEBHOOK_URL:
#         requests.post(SLACK_WEBHOOK_URL, json={"text": summary})
#         print("📤 Weekly summary sent to Slack.")
#     else:
#         print(summary)

#  -------------------------------------------------------------------------------------------

# def send_weekly_summary():
#     print("📊 Weekly summary task running...")

# try:
#     for job in scheduler.get_jobs():
#         if job.id == "weekly_summary_job":
#             scheduler.cancel(job)
#     scheduler.cron(
#         "0 9 * * MON",
#         func=send_weekly_summary,
#         queue_name="monday",
#         id="weekly_summary_job",
#     )
#     print("🕒 Weekly summary job scheduled (Mondays 9 AM IST).")
# except Exception as e:
#     print("⚠️ Could not schedule weekly summary:", e)























# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# import os
# import re
# import json
# import requests
# from datetime import datetime
# from dotenv import load_dotenv
# from redis import Redis
# from rq import Queue
# from rq_scheduler import Scheduler
# from sentence_transformers import SentenceTransformer  # ✅ for semantic embeddings
# from app.storage import VectorStore

# load_dotenv()

# # ------------- Redis / Queue -------------
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# redis_conn = Redis.from_url(REDIS_URL)
# q = Queue("monday", connection=redis_conn)
# scheduler = Scheduler(queue=q, connection=redis_conn)

# # ------------- Config -------------
# MONDAY_API_URL = "https://api.monday.com/v2"
# MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "").strip()
# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
# ALERT_NEGATIVE_THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))

# # ------------- Sentiment Models -------------
# USE_CARDIFFNLP = False
# USE_DISTILBERT = False
# hf_pipeline = None

# try:
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
#     model_name = "cardiffnlp/twitter-roberta-base-sentiment"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     hf_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
#     USE_CARDIFFNLP = True
#     print("✅ Using CardiffNLP/twitter-roberta-base-sentiment model.")
# except Exception as e1:
#     print("⚠️ CardiffNLP unavailable:", e1)
#     try:
#         from transformers import pipeline
#         hf_pipeline = pipeline(
#             "sentiment-analysis",
#             model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#         )
#         USE_DISTILBERT = True
#         print("✅ Using DistilBERT SST-2 fallback model.")
#     except Exception as e2:
#         hf_pipeline = None
#         print("❌ No transformer model available:", e2)

# # ✅ Sentence embedding model
# try:
#     embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     print("✅ Sentence embedding model loaded (all-MiniLM-L6-v2).")
# except Exception as e:
#     embedder = None
#     print("⚠️ Could not load SentenceTransformer:", e)

# # ------------- Keyword Patterns -------------
# NEGATIVE_KEYWORDS = [
#     "bad", "awful", "terrible", "worst", "fail", "failure", "wrong", "problem",
#     "error", "issue", "delay", "hate", "not good", "not working", "stuck",
#     "broken", "poor", "disappointed", "unhappy", "hard", "bug", "slow",
#     "crash", "horrible", "sucks", "waste", "not satisfied", "useless"
# ]
# POSITIVE_KEYWORDS = [
#     "good", "great", "awesome", "amazing", "excellent", "love", "fantastic",
#     "perfect", "well done", "nice", "superb", "wonderful", "brilliant", "cool",
#     "outstanding", "happy", "impressive"
# ]

# _negative_patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in NEGATIVE_KEYWORDS]
# _positive_patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in POSITIVE_KEYWORDS]

# # ------------- Hybrid Sentiment Detection -------------
# def predict_sentiment_combined(text):
#     text_str = (text or "").strip()
#     if not text_str:
#         return {"label": "NEUTRAL", "score": 0.0}

#     # 🧠 1️⃣ Keyword override
#     neg_match = any(p.search(text_str) for p in _negative_patterns)
#     pos_match = any(p.search(text_str) for p in _positive_patterns)
#     if neg_match and not pos_match:
#         print(f"🧠 Keyword override: NEGATIVE → '{text_str}'")
#         return {"label": "NEGATIVE", "score": 0.995}
#     elif pos_match and not neg_match:
#         print(f"🧠 Keyword override: POSITIVE → '{text_str}'")
#         return {"label": "POSITIVE", "score": 0.995}

#     # 🤖 2️⃣ Transformer fallback
#     if hf_pipeline is None:
#         print("⚠️ No model loaded — defaulting to NEUTRAL.")
#         return {"label": "NEUTRAL", "score": 0.0}
#     try:
#         out = hf_pipeline(text_str, truncation=True)
#         result = out[0]
#         label_raw = str(result.get("label", "")).upper()
#         score = float(result.get("score", 0.0))

#         # Normalize Cardiff labels
#         if "LABEL" in label_raw:
#             if label_raw.endswith("0"):
#                 label = "NEGATIVE"
#             elif label_raw.endswith("1"):
#                 label = "NEUTRAL"
#             elif label_raw.endswith("2"):
#                 label = "POSITIVE"
#             else:
#                 label = "NEUTRAL"
#         else:
#             label = label_raw if label_raw in ("NEGATIVE", "POSITIVE", "NEUTRAL") else "NEUTRAL"

#         print(f"🤖 Model ({'Cardiff' if USE_CARDIFFNLP else 'DistilBERT'}) output: {label_raw} → {label} ({score:.2f})")

#         if score < 0.55:
#             return {"label": "NEUTRAL", "score": score}
#         return {"label": label, "score": score}
#     except Exception as e:
#         print("⚠️ Sentiment model error:", e)
#         return {"label": "NEUTRAL", "score": 0.0}

# # ------------- Monday API Helper -------------
# def fetch_monday_item_details(board_id, item_id):
#     if not MONDAY_API_TOKEN:
#         print("⚠️ MONDAY_API_TOKEN not set; cannot fetch item details.")
#         return None
#     headers = {"Authorization": MONDAY_API_TOKEN}
#     query = """
#     query ($board_id: [Int], $item_id: [Int]) {
#       boards(ids: $board_id) {
#         name
#         items (ids: $item_id) {
#           name
#           creator {
#             name
#             email
#           }
#         }
#       }
#     }
#     """
#     variables = {"board_id": [int(board_id)], "item_id": [int(item_id)]}
#     try:
#         resp = requests.post(MONDAY_API_URL, headers=headers,
#                              json={"query": query, "variables": variables}, timeout=10)
#         data = resp.json()
#         board = data["data"]["boards"][0]
#         item = board["items"][0]
#         creator = item.get("creator") or {}
#         return {
#             "board_name": board.get("name", "N/A"),
#             "item_name": item.get("name", "N/A"),
#             "creator_name": creator.get("name", "Unknown"),
#             "creator_email": creator.get("email", "-"),
#         }
#     except Exception as e:
#         print("⚠️ Failed to fetch item details:", e)
#         try:
#             print("🧾 [DEBUG] Monday API raw response:", resp.text)
#         except:
#             pass
        

# # ------------- Queue Function -------------
# def enqueue_monday_event(event: dict):
#     print("📩 [API] Received monday event, queueing job...")
#     try:
#         q.enqueue("app.tasks.process_event", event)
#     except Exception:
#         q.enqueue(process_event, event)

# # ------------- VectorStore Helper -------------
# def add_vector_to_store(vec, meta):
#     try:
#         vs = VectorStore()
#         vs.add_vector(vec, meta)
#         print("💾 [VectorStore] Added vector successfully.")
#     except Exception as e:
#         print("⚠️ [VectorStore] Failed to add vector:", e)


# # ------------- Worker Logic -------------
# def process_event(event: dict):
#     print("🟢 [Worker] Processing event:", event)
#     try:
#         evt = event.get("event", {})
#         board_id = evt.get("boardId")
#         item_id = evt.get("pulseId") or evt.get("itemId")

#         # ✅ Try fetching actual task title from Monday API
#         details = fetch_monday_item_details(board_id, item_id)
#         if details and details.get("item_name") and details["item_name"].lower() != "new task":
#             text = details["item_name"]
#             print(f"🧾 [DEBUG] Using Monday API-fetched text: '{text}'")
#         else:
#             # fallback
#             text = evt.get("pulseName") or evt.get("itemName") or evt.get("text") or json.dumps(evt)[:2000]
#             print(f"🧾 [DEBUG] Extracted text for sentiment: '{text}'")

#         # 🧠 Hybrid Sentiment
#         sentiment = predict_sentiment_combined(text)
#         label = sentiment.get("label", "NEUTRAL").upper()
#         score = float(sentiment.get("score", 0.0))
#         print(f"🔍 Final Sentiment: {label} ({score})")

#         # 💾 Store vector — use embeddings if available
#         if embedder:
#             embedding = embedder.encode(text).tolist()
#         else:
#             embedding = [score]
#         add_vector_to_store(embedding, {"text": text, "sentiment": sentiment})

#         # 🕒 Slack notification
#         board_name = details.get("board_name") if details else "N/A"
#         item_name = details.get("item_name") if details else text[:80]
#         creator_name = details.get("creator_name") if details else "Unknown"
#         creator_email = details.get("creator_email") if details else "-"
#         timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

#         slack_text = (
#             f"*{'🚨 Negative' if label == 'NEGATIVE' else '🎉 Positive'} Sentiment Detected*\n"
#             f"*Board:* {board_name}\n"
#             f"*Item:* {item_name}\n"
#             f"*Created by:* {creator_name} ({creator_email})\n"
#             f"*Sentiment:* `{label}` ({score:.2f})\n"
#             f"*Time:* {timestamp}\n\n"
#             "_Generated automatically by your Monday–Slack AI Monitor 🤖_"
#         )

#         if SLACK_WEBHOOK_URL:
#             if label == "NEGATIVE" and score >= ALERT_NEGATIVE_THRESHOLD:
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🚨 Slack alert sent for NEGATIVE sentiment.")
#             elif label == "POSITIVE":
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🎉 Slack alert sent for POSITIVE sentiment.")
#         else:
#             print("⚠️ SLACK_WEBHOOK_URL not configured.")

#         print("💾 Processing complete.")
#     except Exception as e:
#         print("❌ Error processing event:", e)

# # ------------- Scheduler (Weekly Summary) -------------
# def send_weekly_summary():
#     print("📊 Weekly summary task running...")

# try:
#     for job in scheduler.get_jobs():
#         if job.id == "weekly_summary_job":
#             scheduler.cancel(job)
#     scheduler.cron(
#         "0 9 * * MON",
#         func=send_weekly_summary,
#         queue_name="monday",
#         id="weekly_summary_job",
#     )
#     print("🕒 Weekly summary job scheduled (Mondays 9 AM IST).")
# except Exception as e:
#     print("⚠️ Could not schedule weekly summary:", e)
























# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# import os
# import re
# import json
# import requests
# from datetime import datetime
# from dotenv import load_dotenv
# from redis import Redis
# from rq import Queue
# from rq_scheduler import Scheduler
# from app.models import SentimentEmbedModel
# from app.storage import VectorStore
# from app.alerts import send_slack_alert

# load_dotenv()

# # ------------- Redis / Queue -------------
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# redis_conn = Redis.from_url(REDIS_URL)
# q = Queue("monday", connection=redis_conn)
# scheduler = Scheduler(queue=q, connection=redis_conn)

# # ------------- Config -------------
# MONDAY_API_URL = "https://api.monday.com/v2"
# MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "").strip()
# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
# ALERT_NEGATIVE_THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))

# # ------------- Sentiment Models -------------
# USE_CARDIFFNLP = False
# USE_DISTILBERT = False
# hf_pipeline = None

# try:
#     # Try CardiffNLP first
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
#     model_name = "cardiffnlp/twitter-roberta-base-sentiment"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     hf_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
#     USE_CARDIFFNLP = True
#     print("✅ Using CardiffNLP/twitter-roberta-base-sentiment model.")
# except Exception as e1:
#     print("⚠️ CardiffNLP model unavailable:", e1)
#     try:
#         # Fallback to DistilBERT
#         from transformers import pipeline
#         hf_pipeline = pipeline(
#             "sentiment-analysis",
#             model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#         )
#         USE_DISTILBERT = True
#         print("✅ Using DistilBERT SST-2 fallback model.")
#     except Exception as e2:
#         hf_pipeline = None
#         print("❌ No transformer model available:", e2)


# # ------------- VectorStore Helper -------------
# def add_vector_to_store(vec, meta):
#     try:
#         vs = VectorStore()
#         vs.add_vector(vec, meta)
#         print("💾 [Worker] Added vector to FAISS store")
#     except Exception as e:
#         print("⚠️ [Worker] Failed to add vector to VectorStore:", e)


# # ------------- Hybrid Sentiment Detection -------------
# NEGATIVE_KEYWORDS = [
#     "bad", "awful", "terrible", "worst", "fail", "failure", "wrong", "problem",
#     "error", "issue", "delay", "hate", "not good", "not working", "stuck",
#     "broken", "poor", "disappointed", "unhappy", "hard", "bug", "slow", "crash",
#     "horrible", "sucks", "waste", "not satisfied", "useless"
# ]

# POSITIVE_KEYWORDS = [
#     "good", "great", "awesome", "amazing", "excellent", "love", "fantastic",
#     "perfect", "well done", "nice", "superb", "wonderful", "brilliant", "cool",
#     "outstanding", "happy", "impressive"
# ]

# _negative_patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in NEGATIVE_KEYWORDS]
# _positive_patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in POSITIVE_KEYWORDS]

# def predict_sentiment_combined(text):
#     """
#     Combined keyword + transformer sentiment analysis.
#     Automatically normalizes CardiffNLP & DistilBERT outputs.
#     """
#     text_str = (text or "").strip()
#     if not text_str:
#         return {"label": "NEUTRAL", "score": 0.0}

#     # 1️⃣ Keyword override logic
#     neg_match = any(p.search(text_str) for p in _negative_patterns)
#     pos_match = any(p.search(text_str) for p in _positive_patterns)

#     if neg_match and not pos_match:
#         print(f"🧠 Keyword override: NEGATIVE → '{text_str}'")
#         return {"label": "NEGATIVE", "score": 0.995}
#     elif pos_match and not neg_match:
#         print(f"🧠 Keyword override: POSITIVE → '{text_str}'")
#         return {"label": "POSITIVE", "score": 0.995}

#     # 2️⃣ Transformer Model Fallback
#     if hf_pipeline is None:
#         print("⚠️ No model loaded — defaulting to NEUTRAL.")
#         return {"label": "NEUTRAL", "score": 0.0}

#     try:
#         out = hf_pipeline(text_str, truncation=True)
#         result = out[0]
#         label_raw = str(result.get("label", "")).upper()
#         score = float(result.get("score", 0.0))

#         # Normalize model label formats
#         if "LABEL" in label_raw:
#             if label_raw.endswith("0"):
#                 label = "NEGATIVE"
#             elif label_raw.endswith("1"):
#                 label = "NEUTRAL"
#             elif label_raw.endswith("2"):
#                 label = "POSITIVE"
#             else:
#                 label = "NEUTRAL"
#         elif label_raw in ("NEGATIVE", "POSITIVE", "NEUTRAL"):
#             label = label_raw
#         else:
#             label = "NEUTRAL"

#         print(f"🤖 Model ({'Cardiff' if USE_CARDIFFNLP else 'DistilBERT'}) output: {label_raw} → {label} ({score:.2f})")

#         if score < 0.55:
#             return {"label": "NEUTRAL", "score": score}

#         return {"label": label, "score": score}

#     except Exception as e:
#         print("⚠️ Sentiment model error:", e)
#         return {"label": "NEUTRAL", "score": 0.0}


# # ------------- Monday API Helper -------------
# def fetch_monday_item_details(board_id, item_id):
#     if not MONDAY_API_TOKEN:
#         print("⚠️ MONDAY_API_TOKEN not set; cannot fetch item details.")
#         return None

#     headers = {"Authorization": MONDAY_API_TOKEN}
#     query = """
#     query ($board_id: [Int], $item_id: [Int]) {
#       boards(ids: $board_id) {
#         name
#         items (ids: $item_id) {
#           name
#           creator {
#             name
#             email
#           }
#         }
#       }
#     }
#     """
#     variables = {"board_id": [int(board_id)], "item_id": [int(item_id)]}

#     try:
#         resp = requests.post(
#             MONDAY_API_URL, headers=headers,
#             json={"query": query, "variables": variables}, timeout=10
#         )
#         data = resp.json()
#         board = data["data"]["boards"][0]
#         item = board["items"][0]
#         creator = item.get("creator") or {}
#         return {
#             "board_name": board.get("name", "N/A"),
#             "item_name": item.get("name", "N/A"),
#             "creator_name": creator.get("name", "Unknown"),
#             "creator_email": creator.get("email", "-"),
#         }
#     except Exception as e:
#         print("⚠️ Failed to fetch item details:", e)
#         return None


# # ------------- Queue Function -------------
# def enqueue_monday_event(event: dict):
#     print("📩 [API] Received monday event, queueing job...")
#     try:
#         q.enqueue("app.tasks.process_event", event)
#     except Exception:
#         q.enqueue(process_event, event)


# # ------------- Worker Logic -------------
# def process_event(event: dict):
#     print("🟢 [Worker] Processing event:", event)
#     try:
#         evt = event.get("event", {})
#         text = evt.get("pulseName") or evt.get("itemName") or evt.get("text") or json.dumps(event)[:2000]
#         print(f"🧾 [DEBUG] Extracted text for sentiment: '{text}'")
#         board_id = evt.get("boardId")
#         item_id = evt.get("pulseId") or evt.get("itemId")

#         sentiment = predict_sentiment_combined(text)
#         label = sentiment.get("label", "NEUTRAL").upper()
#         score = float(sentiment.get("score", 0.0))

#         print(f"🔍 Final Sentiment: {label} ({score})")
        


#         # Optional: Vector store
#         add_vector_to_store([score], {"text": text, "sentiment": sentiment})

#         details = fetch_monday_item_details(board_id, item_id)
#         timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

#         board_name = details.get("board_name") if details else "N/A"
#         item_name = details.get("item_name") if details else text[:80]
#         creator_name = details.get("creator_name") if details else "Unknown"
#         creator_email = details.get("creator_email") if details else "-"

#         slack_text = (
#             f"*{'🚨 Negative' if label == 'NEGATIVE' else '🎉 Positive'} Sentiment Detected*\n"
#             f"*Board:* {board_name}\n"
#             f"*Item:* {item_name}\n"
#             f"*Created by:* {creator_name} ({creator_email})\n"
#             f"*Sentiment:* `{label}` ({score:.2f})\n"
#             f"*Time:* {timestamp}\n\n"
#             "_Generated automatically by your Monday–Slack AI Monitor 🤖_"
#         )

#         if SLACK_WEBHOOK_URL:
#             if label == "NEGATIVE" and score >= ALERT_NEGATIVE_THRESHOLD:
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🚨 Slack alert sent for NEGATIVE sentiment.")
#             elif label == "POSITIVE":
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🎉 Slack alert sent for POSITIVE sentiment.")
#         else:
#             print("⚠️ SLACK_WEBHOOK_URL not configured.")

#         print("💾 Processing complete.")
#     except Exception as e:
#         print("❌ Error processing event:", e)


# # ------------- Scheduler (fixed) -------------
# def send_weekly_summary():
#     print("📊 Weekly summary task running...")

# try:
#     for job in scheduler.get_jobs():
#         if job.id == "weekly_summary_job":
#             scheduler.cancel(job)
#     scheduler.cron(
#         "0 9 * * MON",
#         func=send_weekly_summary,
#         queue_name="monday",
#         id="weekly_summary_job",
#     )
#     print("🕒 Weekly summary job scheduled (Mondays 9 AM IST).")
# except Exception as e:
#     print("⚠️ Could not schedule weekly summary:", e)
















# # src/app/tasks.py
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# import os
# import re
# import json
# import requests
# from datetime import datetime
# from dotenv import load_dotenv
# from redis import Redis
# from rq import Queue
# from rq_scheduler import Scheduler
# from app.models import SentimentEmbedModel
# from app.storage import VectorStore
# from app.alerts import send_slack_alert

# load_dotenv()

# # ------------- Redis / Queue -------------
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# redis_conn = Redis.from_url(REDIS_URL)
# q = Queue("monday", connection=redis_conn)
# scheduler = Scheduler(queue=q, connection=redis_conn)

# # ------------- Config -------------
# MONDAY_API_URL = "https://api.monday.com/v2"
# MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "").strip()
# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
# ALERT_NEGATIVE_THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))

# # ------------- Sentiment Models -------------
# USE_CARDIFFNLP = False
# USE_DISTILBERT = False
# hf_pipeline = None

# try:
#     # Try CardiffNLP first
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
#     model_name = "cardiffnlp/twitter-roberta-base-sentiment"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     hf_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
#     USE_CARDIFFNLP = True
#     print("✅ Using CardiffNLP/twitter-roberta-base-sentiment model.")
# except Exception as e1:
#     print("⚠️ CardiffNLP model unavailable:", e1)
#     try:
#         # Fallback to distilbert
#         from transformers import pipeline
#         hf_pipeline = pipeline(
#             "sentiment-analysis",
#             model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#         )
#         USE_DISTILBERT = True
#         print("✅ Using DistilBERT SST-2 fallback model.")
#     except Exception as e2:
#         hf_pipeline = None
#         print("❌ No transformer model available:", e2)


# # ------------- VectorStore Helper -------------
# def add_vector_to_store(vec, meta):
#     try:
#         vs = VectorStore()
#         vs.add_vector(vec, meta)
#         print("💾 [Worker] Added vector to FAISS store")
#     except Exception as e:
#         print("⚠️ [Worker] Failed to add vector to VectorStore:", e)


# # ------------- Hybrid Sentiment Detection -------------
# NEGATIVE_KEYWORDS = [
#     "bad", "awful", "terrible", "worst", "fail", "failure", "wrong", "problem",
#     "error", "issue", "delay", "hate", "not good", "not working", "stuck",
#     "broken", "poor", "disappointed", "unhappy", "hard", "bug", "slow", "crash",
#     "horrible", "sucks", "waste", "not satisfied"
# ]

# _keyword_patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in NEGATIVE_KEYWORDS]
# _phrase_patterns = [re.compile(r"\bnot\s+(good|great|working|helpful)\b", re.IGNORECASE)]

# def predict_sentiment_combined(text):
#     """
#     Combined keyword + transformer sentiment analysis.
#     Automatically normalizes CardiffNLP output.
#     """
#     text_str = (text or "").strip()
#     if not text_str:
#         return {"label": "NEUTRAL", "score": 0.0}

#     # 1️⃣ Keyword override
#     matched = []
#     for p in _keyword_patterns + _phrase_patterns:
#         if p.search(text_str):
#             matched.append(p.pattern)

#     if matched:
#         print(f"🧠 Keyword override triggered for text: '{text_str}'. Matches: {matched}")
#         return {"label": "NEGATIVE", "score": 0.995}

#     # 2️⃣ Transformer Model
#     if hf_pipeline is None:
#         print("⚠️ No model loaded — defaulting to NEUTRAL.")
#         return {"label": "NEUTRAL", "score": 0.0}

#     try:
#         out = hf_pipeline(text_str, truncation=True)
#         result = out[0]

#         # Normalize output across models
#         label_raw = str(result.get("label", "")).upper()
#         score = float(result.get("score", 0.0))

#         # CardiffNLP outputs numeric labels
#         if "LABEL" in label_raw:
#             if label_raw.endswith("0"):
#                 label = "NEGATIVE"
#             elif label_raw.endswith("1"):
#                 label = "NEUTRAL"
#             elif label_raw.endswith("2"):
#                 label = "POSITIVE"
#             else:
#                 label = "NEUTRAL"
#         elif label_raw in ("NEGATIVE", "POSITIVE", "NEUTRAL"):
#             label = label_raw
#         else:
#             label = "NEUTRAL"

#         print(f"🤖 Model ({'Cardiff' if USE_CARDIFFNLP else 'DistilBERT'}) output: {label_raw} → {label} ({score:.2f})")

#         if score < 0.55:
#             return {"label": "NEUTRAL", "score": score}

#         return {"label": label, "score": score}

#     except Exception as e:
#         print("⚠️ Sentiment model error:", e)
#         return {"label": "NEUTRAL", "score": 0.0}


# # ------------- Monday API Helper -------------
# def fetch_monday_item_details(board_id, item_id):
#     if not MONDAY_API_TOKEN:
#         print("⚠️ MONDAY_API_TOKEN not set; cannot fetch item details.")
#         return None

#     headers = {"Authorization": MONDAY_API_TOKEN}
#     query = """
#     query ($board_id: [Int], $item_id: [Int]) {
#       boards(ids: $board_id) {
#         name
#         items (ids: $item_id) {
#           name
#           creator {
#             name
#             email
#           }
#         }
#       }
#     }
#     """
#     variables = {"board_id": [int(board_id)], "item_id": [int(item_id)]}

#     try:
#         resp = requests.post(
#             MONDAY_API_URL, headers=headers,
#             json={"query": query, "variables": variables}, timeout=10
#         )
#         data = resp.json()
#         board = data["data"]["boards"][0]
#         item = board["items"][0]
#         creator = item.get("creator") or {}
#         return {
#             "board_name": board.get("name", "N/A"),
#             "item_name": item.get("name", "N/A"),
#             "creator_name": creator.get("name", "Unknown"),
#             "creator_email": creator.get("email", "-"),
#         }
#     except Exception as e:
#         print("⚠️ Failed to fetch item details:", e)
#         return None


# # ------------- Queue Function -------------
# def enqueue_monday_event(event: dict):
#     print("📩 [API] Received monday event, queueing job...")
#     try:
#         q.enqueue("app.tasks.process_event", event)
#     except Exception:
#         q.enqueue(process_event, event)


# # ------------- Worker Logic -------------
# def process_event(event: dict):
#     print("🟢 [Worker] Processing event:", event)
#     try:
#         evt = event.get("event", {})
#         text = evt.get("pulseName") or evt.get("itemName") or evt.get("text") or json.dumps(event)[:2000]
#         board_id = evt.get("boardId")
#         item_id = evt.get("pulseId") or evt.get("itemId")

#         sentiment = predict_sentiment_combined(text)
#         label = sentiment.get("label", "NEUTRAL").upper()
#         score = float(sentiment.get("score", 0.0))

#         print(f"🔍 Final Sentiment: {label} ({score})")

#         # Optional: Vector store logging
#         add_vector_to_store([score], {"text": text, "sentiment": sentiment})

#         details = fetch_monday_item_details(board_id, item_id)
#         timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

#         board_name = details.get("board_name") if details else "N/A"
#         item_name = details.get("item_name") if details else text[:80]
#         creator_name = details.get("creator_name") if details else "Unknown"
#         creator_email = details.get("creator_email") if details else "-"

#         slack_text = (
#             f"*{'🚨 Negative' if label == 'NEGATIVE' else '🎉 Positive'} Sentiment Detected*\n"
#             f"*Board:* {board_name}\n"
#             f"*Item:* {item_name}\n"
#             f"*Created by:* {creator_name} ({creator_email})\n"
#             f"*Sentiment:* `{label}` ({score:.2f})\n"
#             f"*Time:* {timestamp}\n\n"
#             "_Generated automatically by your Monday–Slack AI Monitor 🤖_"
#         )

#         if SLACK_WEBHOOK_URL:
#             if label == "NEGATIVE" and score >= ALERT_NEGATIVE_THRESHOLD:
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🚨 Slack alert sent for NEGATIVE sentiment.")
#             elif label == "POSITIVE":
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🎉 Slack alert sent for POSITIVE sentiment.")
#         else:
#             print("⚠️ SLACK_WEBHOOK_URL not configured.")

#         print("💾 Processing complete.")
#     except Exception as e:
#         print("❌ Error processing event:", e)


# # ------------- Scheduler (fixed) -------------
# def send_weekly_summary():
#     print("📊 Weekly summary task running...")

# try:
#     for job in scheduler.get_jobs():
#         if job.id == "weekly_summary_job":
#             scheduler.cancel(job)
#     scheduler.cron(
#         "0 9 * * MON",
#         func=send_weekly_summary,
#         queue_name="monday",
#         id="weekly_summary_job",
#     )
#     print("🕒 Weekly summary job scheduled (Mondays 9 AM IST).")
# except Exception as e:
#     print("⚠️ Could not schedule weekly summary:", e)









# # src/app/tasks.py
# import os
# import re
# import json
# import requests
# from datetime import datetime
# from dotenv import load_dotenv
# from redis import Redis
# from rq import Queue
# from rq_scheduler import Scheduler
# from app.models import SentimentEmbedModel
# from app.storage import VectorStore
# from app.alerts import send_slack_alert

# load_dotenv()

# # ------------- Redis / Queue -------------
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# redis_conn = Redis.from_url(REDIS_URL)
# q = Queue("monday", connection=redis_conn)
# scheduler = Scheduler(queue=q, connection=redis_conn)

# # ------------- Monday + Slack config -------------
# MONDAY_API_URL = "https://api.monday.com/v2"
# MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "").strip()
# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
# ALERT_NEGATIVE_THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))

# # ------------- Models -------------
# USE_FALLBACK_PIPELINE = False
# try:
#     from app.models import SentimentEmbedModel

#     def get_model():
#         return SentimentEmbedModel()

#     def analyze_sentiment(model, text):
#         return model.analyze_sentiment(text)

#     def embed_text(model, text):
#         return model.embed_text(text)

# except Exception:
#     USE_FALLBACK_PIPELINE = True
#     try:
#         from transformers import pipeline
#         hf_pipeline = pipeline(
#             "sentiment-analysis",
#             model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#         )
#     except Exception as e:
#         hf_pipeline = None
#         print("⚠️ Warning: no local SentimentEmbedModel and failed loading HF pipeline:", e)

#     def get_model():
#         return None

#     def analyze_sentiment(model, text):
#         if hf_pipeline is None:
#             raise RuntimeError("No sentiment model available")
#         out = hf_pipeline(text, truncation=True)
#         res = out[0]
#         return {"label": res["label"].upper(), "score": float(res["score"])}

#     def embed_text(model, text):
#         return [float(hash(text) % 10_000) / 10_000.0]  # dummy vector


# # ------------- VectorStore helper -------------
# def add_vector_to_store(vec, meta):
#     try:
#         vs = VectorStore()
#         vs.add_vector(vec, meta)
#         print("💾 [Worker] Added vector to FAISS store")
#     except Exception as e:
#         print("⚠️ [Worker] Failed to add vector to VectorStore:", e)


# # ------------- Improved Hybrid Sentiment Detection -------------
# NEGATIVE_KEYWORDS = [
#     "bad", "awful", "terrible", "worst", "fail", "failure", "wrong", "problem",
#     "error", "issue", "delay", "hate", "not good", "not working", "stuck",
#     "broken", "poor", "disappointed", "unhappy", "hard", "bug", "slow",
#     "crash", "horrible", "sucks", "waste", "not satisfied"
# ]

# _keyword_patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in NEGATIVE_KEYWORDS]
# _phrase_patterns = [re.compile(r"\bnot\s+(good|great|working|helpful)\b", re.IGNORECASE)]

# def predict_sentiment_combined(model, text, keyword_override=True):
#     """
#     Combines keyword-based and ML-based sentiment for accurate detection.
#     """
#     text_str = (text or "").strip()
#     if not text_str:
#         return {"label": "NEUTRAL", "score": 0.0}

#     # 1️⃣ Keyword-based detection (regex-based)
#     matched = []
#     for p in _keyword_patterns + _phrase_patterns:
#         if p.search(text_str):
#             matched.append(p.pattern)

#     if matched and keyword_override:
#         print(f"🧠 Keyword override triggered for text: '{text_str}'. Matched patterns: {matched}")
#         return {"label": "NEGATIVE", "score": 0.995}

#     # 2️⃣ ML-based fallback
#     try:
#         out = analyze_sentiment(model, text_str)
#         label_raw = str(out.get("label", "")).upper()
#         score = float(out.get("score", 0.0))

#         if label_raw in ("NEGATIVE", "LABEL_0", "NEG"):
#             label = "NEGATIVE"
#         elif label_raw in ("POSITIVE", "LABEL_2", "POS"):
#             label = "POSITIVE"
#         elif label_raw in ("NEUTRAL", "LABEL_1", "NEU"):
#             label = "NEUTRAL"
#         else:
#             label = "NEUTRAL"

#         print(f"🤖 Model output for '{text_str}': {label_raw} ({score}) → normalized: {label}")
#         if score < 0.55:
#             return {"label": "NEUTRAL", "score": score}

#         return {"label": label, "score": score}

#     except Exception as e:
#         print("⚠️ Sentiment analysis error:", e)
#         return {"label": "NEUTRAL", "score": 0.0}


# # ------------- Monday API helper -------------
# def fetch_monday_item_details(board_id, item_id):
#     if not MONDAY_API_TOKEN:
#         print("⚠️ [Worker] MONDAY_API_TOKEN not set; cannot fetch item details.")
#         return None

#     headers = {"Authorization": MONDAY_API_TOKEN}
#     query = """
#     query ($board_id: [Int], $item_id: [Int]) {
#       boards(ids: $board_id) {
#         name
#         items (ids: $item_id) {
#           name
#           creator {
#             name
#             email
#           }
#         }
#       }
#     }
#     """
#     variables = {"board_id": [int(board_id)], "item_id": [int(item_id)]}

#     try:
#         resp = requests.post(
#             MONDAY_API_URL, headers=headers,
#             json={"query": query, "variables": variables}, timeout=10
#         )
#         data = resp.json()
#         board = data["data"]["boards"][0]
#         item = board["items"][0]
#         creator = item.get("creator") or {}
#         return {
#             "board_name": board.get("name", "N/A"),
#             "item_name": item.get("name", "N/A"),
#             "creator_name": creator.get("name", "Unknown"),
#             "creator_email": creator.get("email", "-"),
#         }
#     except Exception as e:
#         print("⚠️ [Worker] Failed to fetch item details:", e)
#         return None


# # ------------- Queue Function -------------
# def enqueue_monday_event(event: dict):
#     print("📩 [API] Received monday event, queueing job...")
#     try:
#         q.enqueue("app.tasks.process_event", event)
#     except Exception:
#         q.enqueue(process_event, event)


# # ------------- Worker Logic -------------
# def process_event(event: dict):
#     print("🟢 [Worker] Processing event:", event)
#     try:
#         evt = event.get("event", {})
#         text = evt.get("pulseName") or evt.get("itemName") or evt.get("text") or json.dumps(event)[:2000]
#         board_id = evt.get("boardId")
#         item_id = evt.get("pulseId") or evt.get("itemId")

#         model = get_model()
#         sentiment = predict_sentiment_combined(model, text)
#         label = sentiment.get("label", "NEUTRAL").upper()
#         score = float(sentiment.get("score", 0.0))

#         print(f"🔍 [Worker] Final Sentiment: {label} ({score})")

#         embedding = embed_text(model, text)
#         add_vector_to_store(embedding, {"text": text, "sentiment": sentiment})

#         details = fetch_monday_item_details(board_id, item_id)
#         timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

#         board_name = details.get("board_name") if details else "N/A"
#         item_name = details.get("item_name") if details else text[:80]
#         creator_name = details.get("creator_name") if details else "Unknown"
#         creator_email = details.get("creator_email") if details else "-"

#         slack_text = (
#             f"*{'🚨 Negative' if label == 'NEGATIVE' else '🎉 Positive'} Sentiment Detected*\n"
#             f"*Board:* {board_name}\n"
#             f"*Item:* {item_name}\n"
#             f"*Created by:* {creator_name} ({creator_email})\n"
#             f"*Sentiment:* `{label}` ({score:.2f})\n"
#             f"*Time:* {timestamp}\n\n"
#             "_Generated automatically by your Monday–Slack AI Monitor 🤖_"
#         )

#         if SLACK_WEBHOOK_URL:
#             if label == "NEGATIVE" and score >= ALERT_NEGATIVE_THRESHOLD:
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🚨 [Worker] Negative sentiment alert sent to Slack.")
#             elif label == "POSITIVE":
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🎉 [Worker] Positive sentiment message sent to Slack.")
#         else:
#             print("⚠️ [Worker] No Slack webhook configured.")

#         print("💾 [Worker] Processing complete.")
#     except Exception as e:
#         print("❌ [Worker] Error processing event:", e)


# # ------------- Scheduler (fixed) -------------
# def send_weekly_summary():
#     """Example weekly summary task"""
#     print("📊 Weekly summary task running... (add your logic here)")

# try:
#     for job in scheduler.get_jobs():
#         if job.id == "weekly_summary_job":
#             scheduler.cancel(job)
#     scheduler.cron(
#         "0 9 * * MON",  # every Monday at 9 AM
#         func=send_weekly_summary,
#         queue_name="monday",
#         id="weekly_summary_job",
#     )
#     print("🕒 Weekly summary job scheduled (Mondays 9 AM IST).")
# except Exception as e:
#     print("⚠️ Could not schedule weekly summary:", e)



















# # src/app/tasks.py
# import os
# import json
# import requests
# from datetime import datetime
# from dotenv import load_dotenv
# from redis import Redis
# from rq import Queue
# from rq_scheduler import Scheduler
# from app.models import SentimentEmbedModel
# from app.storage import VectorStore
# from app.alerts import send_slack_alert

# load_dotenv()

# # ------------- Redis / Queue -------------
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# redis_conn = Redis.from_url(REDIS_URL)
# q = Queue("monday", connection=redis_conn)
# scheduler = Scheduler(queue=q, connection=redis_conn)

# # ------------- Monday + Slack config -------------
# MONDAY_API_URL = "https://api.monday.com/v2"
# MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "").strip()
# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
# ALERT_NEGATIVE_THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))

# # ------------- Models -------------
# USE_FALLBACK_PIPELINE = False
# try:
#     from app.models import SentimentEmbedModel

#     def get_model():
#         return SentimentEmbedModel()

#     def analyze_sentiment(model, text):
#         return model.analyze_sentiment(text)

#     def embed_text(model, text):
#         return model.embed_text(text)

# except Exception:
#     USE_FALLBACK_PIPELINE = True
#     try:
#         from transformers import pipeline
#         hf_pipeline = pipeline(
#             "sentiment-analysis",
#             model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#         )
#     except Exception as e:
#         hf_pipeline = None
#         print("⚠️ Warning: no local SentimentEmbedModel and failed loading HF pipeline:", e)

#     def get_model():
#         return None

#     def analyze_sentiment(model, text):
#         if hf_pipeline is None:
#             raise RuntimeError("No sentiment model available")
#         out = hf_pipeline(text, truncation=True)
#         res = out[0]
#         return {"label": res["label"].upper(), "score": float(res["score"])}

#     def embed_text(model, text):
#         return [float(hash(text) % 10_000) / 10_000.0]  # dummy vector

# # ------------- VectorStore helper -------------
# def add_vector_to_store(vec, meta):
#     try:
#         vs = VectorStore()
#         vs.add_vector(vec, meta)
#         print("💾 [Worker] Added vector to FAISS store")
#     except Exception as e:
#         print("⚠️ [Worker] Failed to add vector to VectorStore:", e)

# # ------------- Negative keyword override -------------
# NEGATIVE_KEYWORDS = [
#     "bad", "awful", "terrible", "worst", "fail", "failure", "wrong", "problem",
#     "error", "issue", "delay", "hate", "not good", "not working", "stuck",
#     "broken", "poor", "disappointed", "unhappy", "hard", "bug", "slow", "crash"
# ]

# def predict_sentiment_combined(model, text, keyword_override=True):
#     """Combine keyword detection and ML model for better accuracy."""
#     text_lower = text.lower().strip()

#     # 🔍 Keyword-based detection first
#     if keyword_override:
#         for kw in NEGATIVE_KEYWORDS:
#             if kw in text_lower:
#                 print(f"🧠 Keyword trigger: '{kw}' detected → NEGATIVE")
#                 return {"label": "NEGATIVE", "score": 0.99}

#     # 🤖 Fallback to model-based analysis
#     try:
#         out = analyze_sentiment(model, text)
#         label = out.get("label", "").upper()
#         score = float(out.get("score", 0.0))

#         if label in ["LABEL_0", "NEG"]:
#             label = "NEGATIVE"
#         elif label in ["LABEL_2", "POS"]:
#             label = "POSITIVE"
#         elif label in ["LABEL_1", "NEU", "NEUTRAL"]:
#             label = "NEUTRAL"

#         if score < 0.6:
#             return {"label": "NEUTRAL", "score": score}

#         return {"label": label, "score": score}

#     except Exception as e:
#         print("⚠️ Sentiment analysis error:", e)
#         return {"label": "NEUTRAL", "score": 0.0}

# # ------------- Monday API helper -------------
# def fetch_monday_item_details(board_id, item_id):
#     if not MONDAY_API_TOKEN:
#         print("⚠️ [Worker] MONDAY_API_TOKEN not set; cannot fetch item details.")
#         return None

#     headers = {"Authorization": MONDAY_API_TOKEN}
#     query = """
#     query ($board_id: [Int], $item_id: [Int]) {
#       boards(ids: $board_id) {
#         name
#         items (ids: $item_id) {
#           name
#           creator {
#             name
#             email
#           }
#         }
#       }
#     }
#     """
#     variables = {"board_id": [int(board_id)], "item_id": [int(item_id)]}

#     try:
#         resp = requests.post(
#             MONDAY_API_URL, headers=headers,
#             json={"query": query, "variables": variables}, timeout=10
#         )
#         data = resp.json()
#         board = data["data"]["boards"][0]
#         item = board["items"][0]
#         creator = item.get("creator") or {}
#         return {
#             "board_name": board.get("name", "N/A"),
#             "item_name": item.get("name", "N/A"),
#             "creator_name": creator.get("name", "Unknown"),
#             "creator_email": creator.get("email", "-"),
#         }
#     except Exception as e:
#         print("⚠️ [Worker] Failed to fetch item details:", e)
#         return None

# # ------------- Queue Function -------------
# def enqueue_monday_event(event: dict):
#     print("📩 [API] Received monday event, queueing job...")
#     try:
#         q.enqueue("app.tasks.process_event", event)
#     except Exception:
#         q.enqueue(process_event, event)

# # ------------- Worker Logic -------------
# def process_event(event: dict):
#     print("🟢 [Worker] Processing event:", event)
#     try:
#         evt = event.get("event", {})
#         text = evt.get("pulseName") or evt.get("itemName") or evt.get("text") or json.dumps(event)[:2000]
#         board_id = evt.get("boardId")
#         item_id = evt.get("pulseId") or evt.get("itemId")

#         model = get_model()
#         sentiment = predict_sentiment_combined(model, text)
#         label = sentiment.get("label", "NEUTRAL").upper()
#         score = float(sentiment.get("score", 0.0))

#         print(f"🔍 [Worker] Final Sentiment: {label} ({score})")

#         embedding = embed_text(model, text)
#         add_vector_to_store(embedding, {"text": text, "sentiment": sentiment})

#         details = fetch_monday_item_details(board_id, item_id)
#         timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

#         board_name = details.get("board_name") if details else "N/A"
#         item_name = details.get("item_name") if details else text[:80]
#         creator_name = details.get("creator_name") if details else "Unknown"
#         creator_email = details.get("creator_email") if details else "-"

#         slack_text = (
#             f"*{'🚨 Negative' if label == 'NEGATIVE' else '🎉 Positive'} Sentiment Detected*\n"
#             f"*Board:* {board_name}\n"
#             f"*Item:* {item_name}\n"
#             f"*Created by:* {creator_name} ({creator_email})\n"
#             f"*Sentiment:* `{label}` ({score:.2f})\n"
#             f"*Time:* {timestamp}\n\n"
#             "_Generated automatically by your Monday–Slack AI Monitor 🤖_"
#         )

#         if SLACK_WEBHOOK_URL:
#             if label == "NEGATIVE" and score >= ALERT_NEGATIVE_THRESHOLD:
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🚨 [Worker] Negative sentiment alert sent to Slack.")
#             elif label == "POSITIVE":
#                 requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text})
#                 print("🎉 [Worker] Positive sentiment message sent to Slack.")
#         else:
#             print("⚠️ [Worker] No Slack webhook configured.")

#         print("💾 [Worker] Processing complete.")
#     except Exception as e:
#         print("❌ [Worker] Error processing event:", e)

# # ------------- Scheduler (fixed) -------------
# def send_weekly_summary():
#     """Example weekly summary task"""
#     print("📊 Weekly summary task running... (add your logic here)")

# try:
#     # Cancel existing summary job manually
#     for job in scheduler.get_jobs():
#         if job.id == "weekly_summary_job":
#             scheduler.cancel(job)
#     scheduler.cron(
#         "0 9 * * MON",  # every Monday at 9 AM
#         func=send_weekly_summary,
#         queue_name="monday",
#         id="weekly_summary_job",
#     )
#     print("🕒 Weekly summary job scheduled (Mondays 9 AM IST).")
# except Exception as e:
#     print("⚠️ Could not schedule weekly summary:", e)
















# # src/app/tasks.py
# import os
# import json
# import re
# import requests
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
# from redis import Redis
# from rq import Queue
# from rq.job import Job
# from rq_scheduler import Scheduler  # 🆕 for scheduling
# from app.models import SentimentEmbedModel
# from app.storage import VectorStore
# from app.alerts import send_slack_alert
# import pytz

# load_dotenv()

# # ------------- Redis / Queue -------------
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# redis_conn = Redis.from_url(REDIS_URL)
# q = Queue("monday", connection=redis_conn)
# scheduler = Scheduler(queue=q, connection=redis_conn)  # 🆕

# # ------------- Monday + Slack config -------------
# MONDAY_API_URL = "https://api.monday.com/v2"
# MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "").strip()
# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
# ALERT_NEGATIVE_THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))
# MONDAY_BASE_URL = "https://view.monday.com/boards"

# # ------------- Negative keyword rules -------------
# NEGATIVE_KEYWORDS = [
#     "hate", "awful", "terrible", "worst", "fail", "failure", "bad idea",
#     "stupid", "sucks", "disappointed", "problem", "broken", "delay",
#     "angry", "frustrat", "unhappy", "not working", "bug"
# ]
# _NEG_K_PATTERNS = [re.compile(r'\b' + re.escape(k) + r'\b', flags=re.I) for k in NEGATIVE_KEYWORDS]

# def contains_negative_keyword(text: str) -> bool:
#     if not text:
#         return False
#     for p in _NEG_K_PATTERNS:
#         if p.search(text):
#             return True
#     return False

# # ------------- Model setup -------------
# USE_FALLBACK_PIPELINE = False
# try:
#     from app.models import SentimentEmbedModel

#     def get_model():
#         return SentimentEmbedModel()

#     def analyze_sentiment(model, text):
#         return model.analyze_sentiment(text)

#     def embed_text(model, text):
#         return model.embed_text(text)

# except Exception:
#     USE_FALLBACK_PIPELINE = True
#     try:
#         from transformers import pipeline
#         hf_pipeline = pipeline(
#             "sentiment-analysis",
#             model="cardiffnlp/twitter-roberta-base-sentiment"
#         )
#         print("✅ Loaded fallback HF model: cardiffnlp/twitter-roberta-base-sentiment")
#     except Exception as e:
#         hf_pipeline = None
#         print("⚠️ Warning: could not load HF pipeline:", e)

#     def get_model():
#         return None

#     def analyze_sentiment(model, text):
#         if hf_pipeline is None:
#             raise RuntimeError("No sentiment model available")
#         out = hf_pipeline(text, truncation=True)
#         res = out[0]
#         lab = res.get("label", "").upper()
#         sc = float(res.get("score", 0.0))
#         mapping = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
#         lab = mapping.get(lab, lab)
#         return {"label": lab, "score": sc}

#     def embed_text(model, text):
#         return [float(hash(text) % 10_000) / 10_000.0]

# # ------------- Combined Sentiment Function -------------
# def predict_sentiment_combined(model, text, keyword_override=True):
#     text_short = (text or "").strip()
#     if keyword_override and contains_negative_keyword(text_short):
#         return {"label": "NEGATIVE", "score": 0.99}
#     try:
#         out = analyze_sentiment(model, text_short)
#         lab = out.get("label", "").upper()
#         sc = float(out.get("score", 0.0))
#         if sc < 0.55:
#             return {"label": "NEUTRAL", "score": sc}
#         return {"label": lab, "score": sc}
#     except Exception as e:
#         print("⚠️ Sentiment model error:", e)
#         return {"label": "NEUTRAL", "score": 0.0}

# # ------------- Optional VectorStore -------------
# try:
#     from app.storage import VectorStore
# except Exception:
#     VectorStore = None
#     print("⚠️ VectorStore not found — skipping FAISS persistence.")

# def add_vector_to_store(vec, meta):
#     if VectorStore:
#         try:
#             vs = VectorStore()
#             vs.add_vector(vec, meta)
#             print("💾 [Worker] Added vector to FAISS store")
#         except Exception as e:
#             print("⚠️ [Worker] Failed to add vector:", e)
#     else:
#         print("ℹ️ [Worker] No VectorStore configured; skipping FAISS storage.")

# # ------------- Monday Item Details -------------
# def fetch_monday_item_details(board_id, item_id):
#     if not MONDAY_API_TOKEN:
#         print("⚠️ [Worker] MONDAY_API_TOKEN not set.")
#         return None
#     headers = {"Authorization": MONDAY_API_TOKEN}
#     query = """
#     query ($board_id: [Int], $item_id: [Int]) {
#       boards(ids: $board_id) {
#         name
#         items (ids: $item_id) {
#           name
#           creator {
#             name
#             email
#           }
#         }
#       }
#     }
#     """
#     variables = {"board_id": [int(board_id)], "item_id": [int(item_id)]}
#     try:
#         resp = requests.post(MONDAY_API_URL, headers=headers, json={"query": query, "variables": variables}, timeout=10)
#         resp.raise_for_status()
#         data = resp.json()
#         board = data.get("data", {}).get("boards", [{}])[0]
#         item = board.get("items", [{}])[0]
#         creator = item.get("creator", {}) or {}
#         return {
#             "board_name": board.get("name", "N/A"),
#             "item_name": item.get("name", "N/A"),
#             "creator_name": creator.get("name", "Unknown"),
#             "creator_email": creator.get("email", "-"),
#         }
#     except Exception as e:
#         print("⚠️ [Worker] Failed to fetch item details:", e)
#         return None

# # ------------- 🆕 WEEKLY SUMMARY TRACKER -------------
# def increment_sentiment(label: str):
#     key = f"sentiment:{label.lower()}"
#     redis_conn.incr(key)

# def get_weekly_summary():
#     return {
#         "positive": int(redis_conn.get("sentiment:positive") or 0),
#         "neutral": int(redis_conn.get("sentiment:neutral") or 0),
#         "negative": int(redis_conn.get("sentiment:negative") or 0),
#     }

# def reset_weekly_counters():
#     redis_conn.delete("sentiment:positive", "sentiment:neutral", "sentiment:negative")

# def send_weekly_summary():
#     """Send a weekly Slack summary of all sentiments."""
#     summary = get_weekly_summary()
#     total = sum(summary.values())
#     timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")

#     if total == 0:
#         print("📊 [Summary] No sentiment data for this week — skipping report.")
#         return

#     summary_text = (
#         f"📊 *Weekly Sentiment Summary*\n"
#         f"- Positive: {summary['positive']}\n"
#         f"- Neutral: {summary['neutral']}\n"
#         f"- Negative: {summary['negative']} 🚨\n"
#         f"_Generated automatically by your Monday–Slack AI Monitor 🤖_\n\n"
#         f"🕒 *Time:* {timestamp}"
#     )

#     if SLACK_WEBHOOK_URL:
#         requests.post(SLACK_WEBHOOK_URL, json={"text": summary_text})
#         print("✅ [Summary] Weekly report sent to Slack.")
#     else:
#         print("⚠️ [Summary] SLACK_WEBHOOK_URL not configured — skipping Slack post.")

#     reset_weekly_counters()

# # Schedule the summary for every Monday at 9 AM IST (once per week)
# def schedule_weekly_summary():
#     try:
#         jobs = scheduler.get_jobs()
#         if not any("send_weekly_summary" in str(j.func_name) for j in jobs):
#             scheduler.cron(
#                 "0 9 * * MON",  # 9 AM every Monday
#                 func=send_weekly_summary,
#                 queue_name="monday",
#                 id="weekly_summary_job",
#                 replace_existing=True,
#             )
#             print("🕒 Weekly summary job scheduled (Mondays 9 AM IST).")
#     except Exception as e:
#         print("⚠️ Could not schedule weekly summary:", e)

# schedule_weekly_summary()

# # ------------- Enqueue -------------
# def enqueue_monday_event(event: dict):
#     print("📩 [API] Received monday event, queueing job...")
#     try:
#         q.enqueue("app.tasks.process_event", event)
#     except Exception:
#         q.enqueue(process_event, event)

# # ------------- Main Worker -------------
# def process_event(event: dict):
#     print("🟢 [Worker] Processing event:", event)
#     try:
#         if isinstance(event, dict) and "event" in event:
#             evt = event["event"]
#             text = evt.get("pulseName") or evt.get("itemName") or evt.get("text")
#             board_id = evt.get("boardId")
#             item_id = evt.get("pulseId") or evt.get("itemId")
#         else:
#             text = event.get("text") if isinstance(event, dict) else str(event)
#             board_id = None
#             item_id = None

#         if not text:
#             text = json.dumps(event)[:2000]

#         model = get_model()
#         sentiment = predict_sentiment_combined(model, text)
#         label = sentiment["label"].upper()
#         score = float(sentiment["score"])
#         print(f"🔍 [Worker] Sentiment result: {sentiment}")

#         increment_sentiment(label)  # 🆕

#         try:
#             embedding = embed_text(model, text) if model else embed_text(None, text)
#         except Exception as e:
#             print("⚠️ [Worker] Embedding failed:", e)
#             embedding = embed_text(None, text)

#         add_vector_to_store(embedding, {"text": text, "sentiment": sentiment})

#         details = fetch_monday_item_details(board_id, item_id) if board_id and item_id else None
#         timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
#         board_name = details.get("board_name") if details else ("Board " + str(board_id))
#         item_name = details.get("item_name") if details else (text[:80] + ("..." if len(text) > 80 else ""))

#         # --- Send Slack Alert only for Negative ---
#         if label == "NEGATIVE" and score >= ALERT_NEGATIVE_THRESHOLD:
#             slack_payload = {"text": f"🚨 Negative sentiment detected for *{item_name}* ({score:.2f})"}
#             if SLACK_WEBHOOK_URL:
#                 resp = requests.post(SLACK_WEBHOOK_URL, json=slack_payload, timeout=10)
#                 print(f"✅ [Worker] Slack NEGATIVE alert sent (status={resp.status_code})")
#             else:
#                 print("⚠️ [Worker] SLACK_WEBHOOK_URL not configured.")
#         else:
#             print("🙂 [Worker] Sentiment not negative — no Slack alert sent.")

#         print("💾 [Worker] Processing complete.")

#     except Exception as e:
#         print("❌ [Worker] Error processing event:", e)













# # src/app/tasks.py
# # src/app/tasks.py
# import os
# import json
# import re
# import requests
# from datetime import datetime
# from dotenv import load_dotenv
# from redis import Redis
# from rq import Queue
# from app.models import SentimentEmbedModel
# from app.storage import VectorStore
# from app.alerts import send_slack_alert

# load_dotenv()

# # ------------- Redis / Queue -------------
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# redis_conn = Redis.from_url(REDIS_URL)
# q = Queue("monday", connection=redis_conn)

# # ------------- Monday + Slack config -------------
# MONDAY_API_URL = "https://api.monday.com/v2"
# MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "").strip()
# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
# ALERT_NEGATIVE_THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))
# MONDAY_BASE_URL = "https://view.monday.com/boards"

# # ------------- Negative keyword rules -------------
# NEGATIVE_KEYWORDS = [
#     "hate", "awful", "terrible", "worst", "fail", "failure", "bad idea",
#     "stupid", "sucks", "disappointed", "problem", "broken", "delay",
#     "angry", "frustrat", "unhappy", "not working", "bug"
# ]
# _NEG_K_PATTERNS = [re.compile(r'\b' + re.escape(k) + r'\b', flags=re.I) for k in NEGATIVE_KEYWORDS]

# def contains_negative_keyword(text: str) -> bool:
#     if not text:
#         return False
#     for p in _NEG_K_PATTERNS:
#         if p.search(text):
#             return True
#     return False

# # ------------- Model setup -------------
# USE_FALLBACK_PIPELINE = False
# try:
#     from app.models import SentimentEmbedModel

#     def get_model():
#         return SentimentEmbedModel()

#     def analyze_sentiment(model, text):
#         return model.analyze_sentiment(text)

#     def embed_text(model, text):
#         return model.embed_text(text)

# except Exception:
#     USE_FALLBACK_PIPELINE = True
#     try:
#         from transformers import pipeline
#         hf_pipeline = pipeline(
#             "sentiment-analysis",
#             model="cardiffnlp/twitter-roberta-base-sentiment"
#         )
#         print("✅ Loaded fallback HF model: cardiffnlp/twitter-roberta-base-sentiment")
#     except Exception as e:
#         hf_pipeline = None
#         print("⚠️ Warning: could not load HF pipeline:", e)

#     def get_model():
#         return None

#     def analyze_sentiment(model, text):
#         if hf_pipeline is None:
#             raise RuntimeError("No sentiment model available")
#         out = hf_pipeline(text, truncation=True)
#         res = out[0]
#         lab = res.get("label", "").upper()
#         sc = float(res.get("score", 0.0))
#         # Cardiff labels mapping: LABEL_0=NEG, LABEL_1=NEU, LABEL_2=POS
#         mapping = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
#         lab = mapping.get(lab, lab)
#         return {"label": lab, "score": sc}

#     def embed_text(model, text):
#         return [float(hash(text) % 10_000) / 10_000.0]

# # ------------- Combined Sentiment Function -------------
# def predict_sentiment_combined(model, text, keyword_override=True):
#     """
#     Returns dict {"label": "NEGATIVE"/"POSITIVE"/"NEUTRAL", "score": float}
#     Combines keyword rules + model fallback + low confidence neutralization.
#     """
#     text_short = (text or "").strip()

#     # 1) Keyword override (high precision)
#     if keyword_override and contains_negative_keyword(text_short):
#         return {"label": "NEGATIVE", "score": 0.99}

#     # 2) Model prediction
#     try:
#         out = analyze_sentiment(model, text_short)
#         lab = out.get("label", "").upper()
#         sc = float(out.get("score", 0.0))
#         if sc < 0.55:
#             return {"label": "NEUTRAL", "score": sc}
#         return {"label": lab, "score": sc}
#     except Exception as e:
#         print("⚠️ Sentiment model error:", e)
#         return {"label": "NEUTRAL", "score": 0.0}

# # ------------- Optional VectorStore -------------
# try:
#     from app.storage import VectorStore
# except Exception:
#     VectorStore = None
#     print("⚠️ VectorStore not found — skipping FAISS persistence.")

# def add_vector_to_store(vec, meta):
#     if VectorStore:
#         try:
#             vs = VectorStore()
#             vs.add_vector(vec, meta)
#             print("💾 [Worker] Added vector to FAISS store")
#         except Exception as e:
#             print("⚠️ [Worker] Failed to add vector:", e)
#     else:
#         print("ℹ️ [Worker] No VectorStore configured; skipping FAISS storage.")

# # ------------- Monday Item Details -------------
# def fetch_monday_item_details(board_id, item_id):
#     if not MONDAY_API_TOKEN:
#         print("⚠️ [Worker] MONDAY_API_TOKEN not set.")
#         return None

#     headers = {"Authorization": MONDAY_API_TOKEN}
#     query = """
#     query ($board_id: [Int], $item_id: [Int]) {
#       boards(ids: $board_id) {
#         name
#         items (ids: $item_id) {
#           name
#           creator {
#             name
#             email
#           }
#         }
#       }
#     }
#     """
#     variables = {"board_id": [int(board_id)], "item_id": [int(item_id)]}
#     try:
#         resp = requests.post(MONDAY_API_URL, headers=headers, json={"query": query, "variables": variables}, timeout=10)
#         resp.raise_for_status()
#         data = resp.json()
#         board = data.get("data", {}).get("boards", [{}])[0]
#         item = board.get("items", [{}])[0]
#         creator = item.get("creator", {}) or {}
#         return {
#             "board_name": board.get("name", "N/A"),
#             "item_name": item.get("name", "N/A"),
#             "creator_name": creator.get("name", "Unknown"),
#             "creator_email": creator.get("email", "-"),
#         }
#     except Exception as e:
#         print("⚠️ [Worker] Failed to fetch item details:", e)
#         return None

# # ------------- Enqueue -------------
# def enqueue_monday_event(event: dict):
#     print("📩 [API] Received monday event, queueing job...")
#     try:
#         q.enqueue("app.tasks.process_event", event)
#     except Exception:
#         q.enqueue(process_event, event)

# # ------------- Main Worker -------------
# def process_event(event: dict):
#     print("🟢 [Worker] Processing event:", event)
#     try:
#         # Extract text
#         if isinstance(event, dict) and "event" in event:
#             evt = event["event"]
#             text = evt.get("pulseName") or evt.get("itemName") or evt.get("text")
#             board_id = evt.get("boardId")
#             item_id = evt.get("pulseId") or evt.get("itemId")
#         else:
#             text = event.get("text") if isinstance(event, dict) else str(event)
#             board_id = None
#             item_id = None

#         if not text:
#             text = json.dumps(event)[:2000]

#         model = get_model()
#         sentiment = predict_sentiment_combined(model, text)
#         label = sentiment["label"].upper()
#         score = float(sentiment["score"])
#         print(f"🔍 [Worker] Sentiment result: {sentiment}")

#         try:
#             embedding = embed_text(model, text) if model else embed_text(None, text)
#         except Exception as e:
#             print("⚠️ [Worker] Embedding failed:", e)
#             embedding = embed_text(None, text)

#         add_vector_to_store(embedding, {"text": text, "sentiment": sentiment})

#         details = fetch_monday_item_details(board_id, item_id) if board_id and item_id else None
#         timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
#         board_name = details.get("board_name") if details else ("Board " + str(board_id))
#         item_name = details.get("item_name") if details else (text[:80] + ("..." if len(text) > 80 else ""))
#         creator_name = details.get("creator_name") if details else "Unknown"
#         creator_email = details.get("creator_email") if details else "-"
#         item_link = f"{MONDAY_BASE_URL}/{board_id}/pulses/{item_id}" if board_id and item_id else None

#         # --- Slack color logic ---
#         if label == "NEGATIVE":
#             color, emoji, title = "#ff4d4d", "🚨", "Negative Sentiment Detected"
#         elif label == "POSITIVE":
#             color, emoji, title = "#36a64f", "🎉", "Positive Sentiment Detected"
#         else:
#             color, emoji, title = "#e0c341", "ℹ️", "Neutral Sentiment Identified"

#         # --- Slack message payload ---
#         slack_payload = {
#             "attachments": [
#                 {
#                     "color": color,
#                     "blocks": [
#                         {
#                             "type": "header",
#                             "text": {"type": "plain_text", "text": f"{emoji} {title}", "emoji": True}
#                         },
#                         {
#                             "type": "section",
#                             "fields": [
#                                 {"type": "mrkdwn", "text": f"*Board:* {board_name}"},
#                                 {"type": "mrkdwn", "text": f"*Item:* <{item_link}|{item_name}>" if item_link else f"*Item:* {item_name}"},
#                                 {"type": "mrkdwn", "text": f"*Created by:* {creator_name} ({creator_email})"},
#                                 {"type": "mrkdwn", "text": f"*Sentiment:* `{label}` ({score:.2f})"},
#                                 {"type": "mrkdwn", "text": f"*Time:* {timestamp}"}
#                             ]
#                         },
#                         {"type": "section", "text": {"type": "mrkdwn", "text": f"> {text[:500]}"}},
#                         {"type": "context", "elements": [{"type": "mrkdwn", "text": "_Generated automatically by your Monday–Slack AI Monitor 🤖_"}]}
#                     ]
#                 }
#             ]
#         }

#         # --- Send Slack Alert ---
#         if not SLACK_WEBHOOK_URL:
#             print("⚠️ [Worker] SLACK_WEBHOOK_URL not configured.")
#         else:
#             resp = requests.post(SLACK_WEBHOOK_URL, json=slack_payload, timeout=10)
#             print(f"✅ [Worker] Slack alert sent (status={resp.status_code})")

#         print("💾 [Worker] Processing complete.")

#     except Exception as e:
#         print("❌ [Worker] Error processing event:", e)














# # src/app/tasks.py
# import os
# import json
# import requests
# from datetime import datetime
# from dotenv import load_dotenv
# from redis import Redis
# from rq import Queue
# from app.models import SentimentEmbedModel
# from app.storage import VectorStore
# from app.alerts import send_slack_alert

# load_dotenv()

# # ------------- Redis / Queue -------------
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# redis_conn = Redis.from_url(REDIS_URL)
# q = Queue("monday", connection=redis_conn)

# # ------------- Monday + Slack config -------------
# MONDAY_API_URL = "https://api.monday.com/v2"
# MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "").strip()
# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
# ALERT_NEGATIVE_THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))
# MONDAY_BASE_URL = "https://view.monday.com/boards"  # used for direct item link

# # ------------- Models: prefer local SentimentEmbedModel if present -------------
# USE_FALLBACK_PIPELINE = False
# try:
#     from app.models import SentimentEmbedModel

#     def get_model():
#         return SentimentEmbedModel()

#     def analyze_sentiment(model, text):
#         return model.analyze_sentiment(text)

#     def embed_text(model, text):
#         return model.embed_text(text)

# except Exception:
#     USE_FALLBACK_PIPELINE = True
#     try:
#         from transformers import pipeline

#         hf_pipeline = pipeline(
#             "sentiment-analysis",
#             model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#             revision="714eb0f",
#         )
#     except Exception as e:
#         hf_pipeline = None
#         print("⚠️ Warning: no local SentimentEmbedModel and failed loading HF pipeline:", e)

#     def get_model():
#         return None

#     def analyze_sentiment(model, text):
#         if hf_pipeline is None:
#             raise RuntimeError("No sentiment model available")
#         out = hf_pipeline(text, truncation=True)
#         res = out[0]
#         return {"label": res["label"].upper(), "score": float(res["score"])}

#     def embed_text(model, text):
#         return [float(hash(text) % 10_000) / 10_000.0]  # placeholder vector


# # ------------- Optional VectorStore / Alerts modules -------------
# try:
#     from app.storage import VectorStore
# except Exception:
#     VectorStore = None
#     print("⚠️ VectorStore not found — FAISS persistence will be skipped.")


# def add_vector_to_store(vec, meta):
#     if VectorStore:
#         try:
#             vs = VectorStore()
#             vs.add_vector(vec, meta)
#             print("💾 [Worker] Added vector to FAISS store")
#         except Exception as e:
#             print("⚠️ [Worker] Failed to add vector to VectorStore:", e)
#     else:
#         print("ℹ️ [Worker] No VectorStore configured; skipping FAISS storage.")


# # ------------- Helper: fetch Monday item details -------------
# def fetch_monday_item_details(board_id, item_id):
#     if not MONDAY_API_TOKEN:
#         print("⚠️ [Worker] MONDAY_API_TOKEN not set; cannot fetch item details.")
#         return None

#     headers = {"Authorization": MONDAY_API_TOKEN}
#     query = """
#     query ($board_id: [Int], $item_id: [Int]) {
#       boards(ids: $board_id) {
#         name
#         items (ids: $item_id) {
#           name
#           creator {
#             name
#             email
#           }
#         }
#       }
#     }
#     """
#     variables = {"board_id": [int(board_id)], "item_id": [int(item_id)]}

#     try:
#         resp = requests.post(
#             MONDAY_API_URL,
#             headers=headers,
#             json={"query": query, "variables": variables},
#             timeout=10,
#         )
#         resp.raise_for_status()
#         data = resp.json()
#         boards = data.get("data", {}).get("boards") or []
#         if not boards:
#             return None
#         board = boards[0]
#         items = board.get("items") or []
#         if not items:
#             return {"board_name": board.get("name", "N/A")}
#         item = items[0]
#         creator = item.get("creator") or {}
#         return {
#             "board_name": board.get("name", "N/A"),
#             "item_name": item.get("name", "N/A"),
#             "creator_name": creator.get("name", "Unknown"),
#             "creator_email": creator.get("email", "-"),
#         }
#     except Exception as e:
#         print("⚠️ [Worker] Failed to fetch item details from Monday:", e)
#         return None


# # ------------- Enqueue function -------------
# def enqueue_monday_event(event: dict):
#     print("📩 [API] Received monday event, queueing job...")
#     try:
#         q.enqueue("app.tasks.process_event", event)
#     except Exception as e:
#         try:
#             q.enqueue(process_event, event)
#         except Exception as e2:
#             print("❌ Failed to enqueue job:", e, e2)
#             raise


# # ------------- Main worker logic -------------
# def process_event(event: dict):
#     print("🟢 [Worker] Processing event:", event)

#     try:
#         text = None
#         if isinstance(event, dict) and "event" in event:
#             evt = event["event"]
#             text = evt.get("pulseName") or evt.get("itemName") or evt.get("text")
#             board_id = evt.get("boardId")
#             item_id = evt.get("pulseId") or evt.get("itemId")
#         else:
#             text = event.get("text") if isinstance(event, dict) else str(event)
#             board_id = None
#             item_id = None

#         if not text:
#             text = json.dumps(event)[:2000]

#         model = get_model()
#         sentiment = analyze_sentiment(model, text)
#         label = sentiment.get("label", "").upper()
#         score = float(sentiment.get("score", 0.0))
#         print(f"🔍 [Worker] Sentiment result: {sentiment}")

#         try:
#             embedding = embed_text(model, text) if model is not None else embed_text(None, text)
#         except Exception as e:
#             print("⚠️ [Worker] Embedding failed, using fallback:", e)
#             embedding = embed_text(None, text)

#         add_vector_to_store(embedding, {"text": text, "sentiment": sentiment})

#         details = fetch_monday_item_details(board_id, item_id) if board_id and item_id else None

#         timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
#         board_name = details.get("board_name") if details else ("Board " + str(board_id) if board_id else "N/A")
#         item_name = details.get("item_name") if details else (text[:80] + ("..." if len(text) > 80 else ""))
#         creator_name = details.get("creator_name") if details else "Unknown"
#         creator_email = details.get("creator_email") if details else "-"
#         item_link = f"{MONDAY_BASE_URL}/{board_id}/pulses/{item_id}" if board_id and item_id else None

#         # --- 🎨 Slack Color Logic ---
#         if label == "NEGATIVE":
#             color = "#ff4d4d"  # Red
#             emoji = "🚨"
#             title = "Negative Sentiment Detected"
#         elif label == "POSITIVE":
#             color = "#36a64f"  # Green
#             emoji = "🎉"
#             title = "Positive Sentiment Detected"
#         else:
#             color = "#e0c341"  # Yellow
#             emoji = "ℹ️"
#             title = "Neutral Sentiment Identified"

#         # --- 💬 Rich Slack message payload ---
#         slack_payload = {
#             "attachments": [
#                 {
#                     "color": color,
#                     "blocks": [
#                         {
#                             "type": "header",
#                             "text": {"type": "plain_text", "text": f"{emoji} {title}", "emoji": True}
#                         },
#                         {
#                             "type": "section",
#                             "fields": [
#                                 {"type": "mrkdwn", "text": f"*Board:* {board_name}"},
#                                 {"type": "mrkdwn", "text": f"*Item:* <{item_link}|{item_name}>" if item_link else f"*Item:* {item_name}"},
#                                 {"type": "mrkdwn", "text": f"*Created by:* {creator_name} ({creator_email})"},
#                                 {"type": "mrkdwn", "text": f"*Sentiment:* `{label}` ({score:.2f})"},
#                                 {"type": "mrkdwn", "text": f"*Time:* {timestamp}"}
#                             ]
#                         },
#                         {
#                             "type": "section",
#                             "text": {"type": "mrkdwn", "text": f"> {text[:500]}"}
#                         },
#                         {
#                             "type": "context",
#                             "elements": [{"type": "mrkdwn", "text": "_Generated automatically by your Monday–Slack AI Monitor 🤖_"}]
#                         }
#                     ]
#                 }
#             ]
#         }

#         # --- 🚀 Slack send logic ---
#         if not SLACK_WEBHOOK_URL:
#             print("⚠️ [Worker] SLACK_WEBHOOK_URL not configured; skipping alert.")
#         else:
#             # Always send POSITIVE/NEGATIVE/NEUTRAL summaries, not just negatives
#             try:
#                 resp = requests.post(SLACK_WEBHOOK_URL, json=slack_payload, timeout=10)
#                 print(f"✅ [Worker] Slack alert sent (status={resp.status_code})")
#             except Exception as e:
#                 print("❌ [Worker] Slack send failed:", e)

#         print("💾 [Worker] Processing complete.")
#     except Exception as e:
#         print("❌ [Worker] Error processing event:", e)





















# # src/app/tasks.py
# import os
# import json
# import requests
# from datetime import datetime
# from dotenv import load_dotenv
# from redis import Redis
# from rq import Queue
# from app.models import SentimentEmbedModel
# from app.storage import VectorStore
# from app.alerts import send_slack_alert

# load_dotenv()

# # ------------- Redis / Queue -------------
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# redis_conn = Redis.from_url(REDIS_URL)
# q = Queue("monday", connection=redis_conn)

# # ------------- Monday + Slack config -------------
# MONDAY_API_URL = "https://api.monday.com/v2"
# MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "").strip()
# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
# ALERT_NEGATIVE_THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))

# # ------------- Models: prefer local SentimentEmbedModel if present -------------
# USE_FALLBACK_PIPELINE = False
# try:
#     # Prefer your existing in-project model which handles both sentiment + embeddings
#     from app.models import SentimentEmbedModel

#     def get_model():
#         return SentimentEmbedModel()

#     def analyze_sentiment(model, text):
#         # Expecting model.analyze_sentiment -> {"label": "NEGATIVE"/"POSITIVE", "score": 0.98}
#         return model.analyze_sentiment(text)

#     def embed_text(model, text):
#         return model.embed_text(text)

# except Exception:
#     # fallback: use HF pipeline if your SentimentEmbedModel isn't available
#     USE_FALLBACK_PIPELINE = True
#     try:
#         from transformers import pipeline

#         hf_pipeline = pipeline(
#             "sentiment-analysis",
#             model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#             revision="714eb0f",
#         )
#     except Exception as e:
#         hf_pipeline = None
#         print("⚠️ Warning: no local SentimentEmbedModel and failed loading HF pipeline:", e)

#     def get_model():
#         return None

#     def analyze_sentiment(model, text):
#         if hf_pipeline is None:
#             raise RuntimeError("No sentiment model available")
#         out = hf_pipeline(text, truncation=True)
#         # HF returns [{"label": "NEGATIVE"/"POSITIVE", "score": float}]
#         res = out[0]
#         return {"label": res["label"].upper(), "score": float(res["score"])}

#     def embed_text(model, text):
#         # fallback: generate a naive embedding using a hash / not suitable for RAG.
#         # If you have sentence-transformers available, swap here for real embeddings.
#         return [float(hash(text) % 10_000) / 10_000.0]  # very small placeholder vector


# # ------------- Optional VectorStore / Alerts modules (use your implementations if present) -------------
# try:
#     from app.storage import VectorStore
# except Exception:
#     VectorStore = None
#     print("⚠️ VectorStore not found — FAISS persistence will be skipped.")

# def add_vector_to_store(vec, meta):
#     if VectorStore:
#         try:
#             vs = VectorStore()
#             vs.add_vector(vec, meta)
#             print("💾 [Worker] Added vector to FAISS store")
#         except Exception as e:
#             print("⚠️ [Worker] Failed to add vector to VectorStore:", e)
#     else:
#         print("ℹ️ [Worker] No VectorStore configured; skipping FAISS storage.")


# # ------------- Helper: fetch Monday item details -------------
# def fetch_monday_item_details(board_id, item_id):
#     """
#     Returns dict with board_name, item_name, creator_name, creator_email or None on failure.
#     """
#     if not MONDAY_API_TOKEN:
#         print("⚠️ [Worker] MONDAY_API_TOKEN not set; cannot fetch item details.")
#         return None

#     headers = {"Authorization": MONDAY_API_TOKEN}
#     query = """
#     query ($board_id: [Int], $item_id: [Int]) {
#       boards(ids: $board_id) {
#         name
#         items (ids: $item_id) {
#           name
#           creator {
#             name
#             email
#           }
#         }
#       }
#     }
#     """
#     variables = {"board_id": [int(board_id)], "item_id": [int(item_id)]}

#     try:
#         resp = requests.post(
#             MONDAY_API_URL,
#             headers=headers,
#             json={"query": query, "variables": variables},
#             timeout=10,
#         )
#         resp.raise_for_status()
#         data = resp.json()
#         # defensive extraction
#         boards = data.get("data", {}).get("boards") or []
#         if not boards:
#             return None
#         board = boards[0]
#         items = board.get("items") or []
#         if not items:
#             return {"board_name": board.get("name", "N/A")}
#         item = items[0]
#         creator = item.get("creator") or {}
#         return {
#             "board_name": board.get("name", "N/A"),
#             "item_name": item.get("name", "N/A"),
#             "creator_name": creator.get("name", "Unknown"),
#             "creator_email": creator.get("email", "-"),
#         }
#     except Exception as e:
#         print("⚠️ [Worker] Failed to fetch item details from Monday:", e)
#         return None


# # ------------- Enqueue function (keeps same behavior you had) -------------
# def enqueue_monday_event(event: dict):
#     """
#     Enqueue the job. Using the string reference keeps compatibility with RQ workers started
#     as they were before (they import app.tasks).
#     """
#     print("📩 [API] Received monday event, queueing job...")
#     try:
#         q.enqueue("app.tasks.process_event", event)
#     except Exception as e:
#         # fallback: enqueue callable directly
#         try:
#             q.enqueue(process_event, event)
#         except Exception as e2:
#             print("❌ Failed to enqueue job (both string and direct):", e, e2)
#             raise


# # ------------- Main worker logic -------------
# def process_event(event: dict):
#     """
#     Process a Monday.com webhook event, analyze sentiment, store vector, and send Slack alerts for negatives.
#     """
#     print("🟢 [Worker] Processing event:", event)

#     try:
#         # extract text candidates
#         text = None
#         # event may have 'text' for manual test, or nested 'event' from Monday
#         if isinstance(event, dict) and "event" in event:
#             evt = event["event"]
#             # Item name sometimes sent as "pulseName" or "itemName" depending on payload
#             text = evt.get("pulseName") or evt.get("itemName") or evt.get("text")
#             board_id = evt.get("boardId")
#             item_id = evt.get("pulseId") or evt.get("itemId")
#         else:
#             text = event.get("text") if isinstance(event, dict) else str(event)
#             board_id = None
#             item_id = None

#         # fallback to string of event if nothing found
#         if not text:
#             text = json.dumps(event)[:2000]

#         # get model (either SentimentEmbedModel or None for fallback)
#         model = get_model()

#         # analyze sentiment
#         sentiment = analyze_sentiment(model, text)
#         label = sentiment.get("label", "").upper()
#         score = float(sentiment.get("score", 0.0))
#         print(f"🔍 [Worker] Sentiment result: {sentiment}")

#         # create embedding via model if available
#         try:
#             embedding = embed_text(model, text) if model is not None else embed_text(None, text)
#         except Exception as e:
#             print("⚠️ [Worker] Embedding failed, using fallback:", e)
#             embedding = embed_text(None, text)

#         # add vector
#         add_vector_to_store(embedding, {"text": text, "sentiment": sentiment})

#         # fetch details (best-effort)
#         details = fetch_monday_item_details(board_id, item_id) if board_id and item_id else None

#         # Prepare Slack message (pretty)
#         timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
#         board_name = details.get("board_name") if details else ("Board " + str(board_id) if board_id else "N/A")
#         item_name = details.get("item_name") if details else (text[:80] + ("..." if len(text) > 80 else ""))
#         creator_name = details.get("creator_name") if details else "Unknown"
#         creator_email = details.get("creator_email") if details else "-"

#         slack_text = (
#             f"🚨 *Negative Sentiment Detected in monday.com item*\n"
#             f"*Board:* {board_name}\n"
#             f"*Item:* {item_name}\n"
#             f"*Created by:* {creator_name} ({creator_email})\n"
#             f"*Sentiment:* `{label}` ({score:.2f})\n"
#             f"*Time:* {timestamp}\n\n"
#             f"_Automatically analyzed by AI webhook worker._"
#         )

#         # Send Slack only if negative and above threshold
#         if label == "NEGATIVE" and score >= ALERT_NEGATIVE_THRESHOLD:
#             if not SLACK_WEBHOOK_URL:
#                 print("⚠️ [Worker] SLACK_WEBHOOK_URL not configured; not sending Slack alert.")
#             else:
#                 try:
#                     resp = requests.post(SLACK_WEBHOOK_URL, json={"text": slack_text}, timeout=10)
#                     print(f"✅ [Worker] Slack alert sent (status={resp.status_code})")
#                 except Exception as e:
#                     print("❌ [Worker] Slack send failed:", e)
#         else:
#             print("🙂 [Worker] No alert sent (not negative enough or non-negative).")

#         print("💾 [Worker] Processing complete.")
#     except Exception as e:
#         print("❌ [Worker] Error processing event:", e)











# import os
# from redis import Redis
# from rq import Queue
# from app.models import SentimentEmbedModel
# from app.storage import VectorStore
# from app.alerts import send_slack_alert

# redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
# q = Queue("monday", connection=redis_conn)


# def enqueue_monday_event(event: dict):
#     print("📩 [API] Received monday event, queueing job...")
#     q.enqueue("app.tasks.process_event", event)


# def extract_text_from_payload(payload):
#     for key in ("text", "update_text", "body", "message"):
#         if isinstance(payload.get(key), str):
#             return payload.get(key)
#     return str(payload)[:2000]


# def process_event(event: dict):
#     print("🟢 [Worker] Processing event:", event)

#     try:
#         text = extract_text_from_payload(event)
#         if not text:
#             print("⚠️ [Worker] No text found, skipping")
#             return

#         model = SentimentEmbedModel()
#         sentiment = model.analyze_sentiment(text)
#         print("🔍 [Worker] Sentiment result:", sentiment)

#         embedding = model.embed_text(text)
#         vs = VectorStore()
#         vs.add_vector(embedding, {"text": text, "sentiment": sentiment})
#         print("💾 [Worker] Added vector to FAISS store")

#         threshold = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.7))
#         if sentiment["label"].upper() == "NEGATIVE" and sentiment["score"] >= threshold:
#             print("🚨 [Worker] Negative sentiment detected, sending Slack alert...")
#             send_slack_alert(text, sentiment)
#             print("✅ [Worker] Slack alert sent!")
#         else:
#             print("🙂 [Worker] No alert sent (not negative enough).")

#     except Exception as e:
#         print("❌ [Worker] Error processing event:", e)



















# import os
# from redis import Redis
# from rq import Queue
# from app.models import SentimentEmbedModel
# from app.storage import VectorStore
# from app.alerts import send_slack_alert

# redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
# q = Queue("monday", connection=redis_conn)

# def enqueue_monday_event(event: dict):
#     q.enqueue(process_event, event)

# def process_event(event: dict):
#     # simple extraction - customize for monday payload
#     text = extract_text_from_payload(event)
#     if not text:
#         return
#     model = SentimentEmbedModel()
#     sentiment = model.analyze_sentiment(text)
#     embedding = model.embed_text(text)
#     vs = VectorStore()
#     vs.add_vector(embedding, {"text": text, "raw": event, "sentiment": sentiment})
#     # Alert rule
#     if sentiment.get("label") == "NEGATIVE" and sentiment.get("score", 0) >= float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.75)):
#         send_slack_alert(text, sentiment, event)

# # helper
# def extract_text_from_payload(payload):
#     # monday payload structure varies; try common keys
#     # This is a safe, minimal extractor — adapt for your payload
#     for key in ("text", "update_text", "pulse_update_text", "body", "message"):
#         if isinstance(payload.get(key), str):
#             return payload.get(key)
#     # attempt nested search
#     raw = str(payload)
#     return raw[:10000]  # fallback

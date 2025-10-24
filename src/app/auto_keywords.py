# src/app/auto_keywords.py
import re
from collections import Counter
from app.storage import VectorStore

# Simple text cleanup
def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

# Auto-learn keywords based on frequent words in each sentiment group
def learn_keywords(min_freq=2):
    vs = VectorStore()
    all_meta = vs.meta
    pos_words, neg_words = [], []

    for m in all_meta:
        label = m["sentiment"]["label"].upper()
        words = clean_text(m["text"]).split()
        if label == "POSITIVE":
            pos_words.extend(words)
        elif label == "NEGATIVE":
            neg_words.extend(words)

    pos_common = [w for w, c in Counter(pos_words).items() if c >= min_freq and len(w) > 3]
    neg_common = [w for w, c in Counter(neg_words).items() if c >= min_freq and len(w) > 3]

    # Remove overlaps
    pos_common = [w for w in pos_common if w not in neg_common]
    neg_common = [w for w in neg_common if w not in pos_common]

    print("🔍 Auto-learned positive keywords:", pos_common[:15])
    print("🔍 Auto-learned negative keywords:", neg_common[:15])

    return {"positive": pos_common, "negative": neg_common}

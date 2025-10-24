import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from app.tasks import enqueue_monday_event

load_dotenv()

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/webhook/monday")
async def monday_webhook(req: Request):
    """
    Handles incoming webhook events from Monday.com.
    Also responds to the verification challenge when Monday connects the webhook.
    """
    try:
        payload = await req.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # ✅ Step 1: Handle the initial verification challenge
    if "challenge" in payload:
        print("🔐 Monday.com verification challenge received:", payload["challenge"])
        return JSONResponse(content={"challenge": payload["challenge"]})

    # ✅ Step 2: Validate payload
    if not payload:
        raise HTTPException(status_code=400, detail="Empty payload")

    # ✅ Step 3: Process actual webhook event
    print("📩 Monday Webhook Event Received:", payload)
    try:
        enqueue_monday_event(payload)
    except Exception as e:
        print("❌ Failed to enqueue Monday event:", e)
        raise HTTPException(status_code=500, detail="Error processing webhook event")

    return {"ok": True}

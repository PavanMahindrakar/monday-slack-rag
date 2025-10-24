# # src/app/alerts.py
import os
import requests
from datetime import datetime

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN")
MONDAY_API_URL = "https://api.monday.com/v2"


def fetch_monday_item_details(board_id, pulse_id):
    """Fetch board name, item name, and creator info from Monday.com."""
    headers = {"Authorization": MONDAY_API_TOKEN, "Content-Type": "application/json"}
    query = """
    query ($board_id: Int!, $item_id: Int!) {
      boards(ids: [$board_id]) {
        name
        items(ids: [$item_id]) {
          name
          creator {
            name
            email
          }
        }
      }
    }
    """
    variables = {"board_id": board_id, "item_id": pulse_id}
    response = requests.post(
        MONDAY_API_URL,
        headers=headers,
        json={"query": query, "variables": variables}
    )

    try:
        data = response.json()
        board = data["data"]["boards"][0]
        item = board["items"][0]
        return {
            "board_name": board["name"],
            "item_name": item["name"],
            "creator_name": item["creator"]["name"],
            "creator_email": item["creator"]["email"]
        }
    except Exception as e:
        print("⚠️ [Monday] Failed to fetch item details:", e)
        return None


def send_slack_alert(text, sentiment, board_id=None, pulse_id=None):
    """Send rich, context-aware alert to Slack."""
    label = sentiment["label"].upper()
    score = sentiment["score"]

    # Choose alert style based on sentiment
    if label == "NEGATIVE":
        color = "#ff4d4d"
        emoji = "🚨"
        title = "Negative Sentiment Detected"
    elif label == "POSITIVE":
        color = "#36a64f"
        emoji = "🎉"
        title = "Positive Sentiment Detected"
    else:
        color = "#e0c341"
        emoji = "ℹ️"
        title = "Neutral Sentiment Update"

    # Fetch Monday item metadata (optional)
    details = None
    if board_id and pulse_id:
        details = fetch_monday_item_details(board_id, pulse_id)

    board_name = details["board_name"] if details else "N/A"
    item_name = details["item_name"] if details else "N/A"
    creator = f"{details['creator_name']} ({details['creator_email']})" if details else "N/A"

    # Slack Block Kit payload
    payload = {
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": f"{emoji} {title}", "emoji": True}
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Board:* {board_name}"},
                            {"type": "mrkdwn", "text": f"*Item:* {item_name}"},
                            {"type": "mrkdwn", "text": f"*Created by:* {creator}"},
                            {"type": "mrkdwn", "text": f"*Sentiment:* `{label}` ({score:.2f})"},
                            {"type": "mrkdwn", "text": f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
                        ]
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"> {text[:500]}"}
                    },
                    {
                        "type": "context",
                        "elements": [
                            {"type": "mrkdwn", "text": "_Generated automatically by your Monday–Slack AI Monitor 🤖_"}
                        ]
                    }
                ]
            }
        ]
    }

    try:
        requests.post(SLACK_WEBHOOK_URL, json=payload)
        print(f"✅ [Slack] {label} alert sent successfully with board details!")
    except Exception as e:
        print(f"❌ [Slack] Failed to send alert: {e}")












# import os, requests, json

# SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")

# def send_slack_alert(text, sentiment, raw_event=None):
#     if not SLACK_WEBHOOK:
#         print("No SLACK_WEBHOOK_URL configured")
#         return
#     title = ":rotating_light: Negative sentiment detected in monday.com item"
#     payload = {
#         "text": title,
#         "attachments": [
#             {
#                 "fallback": title,
#                 "color": "danger",
#                 "title": "Item with negative sentiment",
#                 "text": text if len(text) < 3000 else text[:3000],
#                 "fields": [
#                     {"title": "Sentiment", "value": f"{sentiment.get('label')} ({sentiment.get('score',0):.2f})", "short": True}
#                 ]
#             }
#         ]
#     }
#     resp = requests.post(SLACK_WEBHOOK, json=payload, timeout=10)
#     if resp.status_code >= 400:
#         print("Failed to send slack alert:", resp.status_code, resp.text)

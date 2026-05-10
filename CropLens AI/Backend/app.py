"""
CropLens — Python/Flask Backend (Google Gemini Free API)
---------------------------------------------------------
Get your FREE API key at: https://aistudio.google.com/app/apikey

Run:
    pip install -r requirements.txt
    set GEMINI_API_KEY=your_key_here        # Windows
    export GEMINI_API_KEY=your_key_here     # Mac/Linux
    python app.py

Then open: http://localhost:5000
"""

import os
import base64
import json
import re
import urllib.request
import urllib.error

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = os.environ.get("GEMINI_API_KEY", "AIzaSyDN9_4YKNuBUO8TfSDK09Q9KJl9rqIfjU8")
MODEL      = "gemini-1.5-flash"          # free tier model
STATIC_DIR = Path(__file__).parent       # index.html lives in the same folder

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app)

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT = """You are an expert botanist and agricultural advisor.
You specialise in these three crops (but can identify others too):
  1. Black Cumin  (Nigella sativa)     - medicinal herb, black seeds
  2. Sweet Pea    (Lathyrus odoratus)  - ornamental flowering plant
  3. Allium Cepa  (Onion)              - bulb vegetable

Look at the image and return ONLY a valid JSON object - no markdown, no extra text - with this exact structure:
{
  "cropName":      "Common name",
  "sciName":       "Scientific name",
  "emoji":         "one relevant emoji",
  "confidence":    85,
  "growingSeason": "e.g. March to June",
  "waterNeeds":    "e.g. Moderate - 300-500 mm/season",
  "timeNeeded":    "e.g. 90-120 days",
  "temperature":   "e.g. 15-25 C optimal",
  "npk":           { "n": 40, "p": 60, "k": 50 },
  "fertRec":       "1-2 sentence fertiliser recommendation",
  "timeline": [
    { "emoji": "🌱", "label": "Sow",       "dur": "Week 1"     },
    { "emoji": "🌿", "label": "Germinate", "dur": "Week 2-3"   },
    { "emoji": "🌸", "label": "Flower",    "dur": "Week 6-8"   },
    { "emoji": "🌾", "label": "Harvest",   "dur": "Week 12-16" }
  ],
  "medicinalTags": ["tag1", "tag2"],
  "medicinalDesc": "Detailed medicinal properties description.",
  "hasMedicinal":  true
}
NPK values are relative percentages 0-100.
If confidence < 40, still give your best guess."""


def _call_gemini(image_b64: str, mime_type: str) -> dict:
    """Send image to Gemini and return parsed JSON dict."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={API_KEY}"
    )

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": image_b64}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        raise RuntimeError(f"Gemini HTTP {e.code}: {err_body}") from e

    # Extract text from Gemini response structure
    try:
        raw = body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response shape: {body}") from exc

    # Strip markdown fences if model adds them
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Body:  { "image": "<base64>", "mediaType": "image/jpeg" }
    Returns: { "success": true,  "data": { ...crop fields... } }
          or { "success": false, "error": "message" }
    """
    if not API_KEY:
        return jsonify(
            success=False,
            error="GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/app/apikey"
        ), 500

    body      = request.get_json(silent=True) or {}
    image_b64 = body.get("image", "").strip()
    mime_type = body.get("mediaType", "image/jpeg")

    if not image_b64:
        return jsonify(success=False, error="No image data received."), 400

    try:
        base64.b64decode(image_b64, validate=True)
    except Exception:
        return jsonify(success=False, error="Invalid base64 image data."), 400

    try:
        data = _call_gemini(image_b64, mime_type)
        return jsonify(success=True, data=data)

    except json.JSONDecodeError as exc:
        return jsonify(success=False, error=f"Model returned malformed JSON: {exc}"), 500
    except RuntimeError as exc:
        return jsonify(success=False, error=str(exc)), 502
    except Exception as exc:
        return jsonify(success=False, error=f"Unexpected error: {exc}"), 500


@app.route("/api/health")
def health():
    return jsonify(status="ok", model=MODEL, api_key_set=bool(API_KEY))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🌿 CropLens server starting -> http://localhost:{port}")
    print(f"   Model  : {MODEL}")
    print(f"   API key: {'SET' if API_KEY else 'NOT SET - get one free at https://aistudio.google.com/app/apikey'}\n")
    app.run(host="0.0.0.0", port=port, debug=False)

import os
import io
import base64
import json
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic
from PIL import Image
import numpy as np

# ─────────────────────────── CONFIG ───────────────────────────
MODEL_PATH   = os.getenv("CROPLENS_MODEL_PATH", "model.keras")   # or "model.h5"
IMG_SIZE     = (224, 224)        # resize target for the CNN
API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS   = 1100
STATIC_DIR   = os.path.dirname(os.path.abspath(__file__))        # folder with index.html

# The three classes your model was trained on — ORDER MUST MATCH your training labels
CLASS_NAMES = [
    "Black Cumin (Nigella sativa)",
    "Sweet Pea (Lathyrus odoratus)",
    "Allium Cepa (Onion)",
]
CLASS_EMOJIS = {
    "Black Cumin (Nigella sativa)":    "🌱",
    "Sweet Pea (Lathyrus odoratus)":   "🌸",
    "Allium Cepa (Onion)":             "🧅",
}

# ─────────────────────────── FLASK APP ────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)  # allow direct calls during local dev

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── MODEL LOADER ─────────────────────
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        log.warning(f"Model file '{MODEL_PATH}' not found — running in Claude-only mode.")
        return
    try:
        # Import TF/Keras lazily so the server still starts without it installed
        try:
            from tensorflow import keras
        except ImportError:
            import keras
        model = keras.models.load_model(MODEL_PATH)
        log.info(f"✅ Model loaded from '{MODEL_PATH}' — running in Model+Claude mode.")
    except Exception as e:
        log.error(f"Failed to load model: {e} — falling back to Claude-only mode.")
        model = None


def predict_with_model(image_bytes: bytes) -> dict:
    """
    Run the loaded CNN on the image.
    Returns {"crop_name": str, "confidence": float (0-100), "sci_name": str}
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0   # normalise to [0,1]
    arr = np.expand_dims(arr, axis=0)               # add batch dim

    probs = model.predict(arr, verbose=0)[0]        # shape: (3,)
    idx   = int(np.argmax(probs))
    confidence = float(probs[idx]) * 100

    crop_name = CLASS_NAMES[idx]
    # Build a minimal sci_name from the class label (text inside parentheses)
    sci_name  = crop_name.split("(")[-1].rstrip(")") if "(" in crop_name else ""
    common    = crop_name.split("(")[0].strip()

    return {
        "crop_name":  common,
        "sci_name":   sci_name,
        "confidence": round(confidence, 1),
        "emoji":      CLASS_EMOJIS.get(crop_name, "🌿"),
    }


# ─────────────────────────── CLAUDE HELPERS ───────────────────
SYSTEM_ENRICH = """You are an expert botanist and agricultural advisor specialising in
Black Cumin (Nigella sativa), Sweet Pea (Lathyrus odoratus), and Allium Cepa (Onion).

Given a pre-identified crop name and confidence, return ONLY a valid JSON object
(no markdown, no explanation) with EXACTLY this structure:

{
  "cropName":     "Common name",
  "sciName":      "Scientific name",
  "emoji":        "single emoji",
  "confidence":   85,
  "growingSeason":"Month range",
  "waterNeeds":   "e.g. Moderate",
  "timeNeeded":   "e.g. 90 days",
  "temperature":  "e.g. 15-25°C",
  "soilType":     "e.g. Sandy loam",
  "sunlight":     "e.g. Full sun",
  "spacing":      "e.g. 15 cm apart",
  "ph":           "e.g. pH 6.5",
  "npk":          {"n":40,"p":60,"k":50},
  "fertRec":      "1-2 sentence recommendation",
  "timeline":     [{"emoji":"🌰","label":"Sow","dur":"Week 1"}],
  "medicinalTags":["tag1","tag2"],
  "medicinalDesc":"Description",
  "hasMedicinal": true
}"""

SYSTEM_VISION = """You are an expert botanist and agricultural advisor specialising in
Black Cumin (Nigella sativa), Sweet Pea (Lathyrus odoratus), and Allium Cepa (Onion).

Identify the plant in the image and return ONLY a valid JSON object
(no markdown, no explanation) with EXACTLY this structure:

{
  "cropName":     "Common name",
  "sciName":      "Scientific name",
  "emoji":        "single emoji",
  "confidence":   85,
  "growingSeason":"Month range",
  "waterNeeds":   "e.g. Moderate",
  "timeNeeded":   "e.g. 90 days",
  "temperature":  "e.g. 15-25°C",
  "soilType":     "e.g. Sandy loam",
  "sunlight":     "e.g. Full sun",
  "spacing":      "e.g. 15 cm apart",
  "ph":           "e.g. pH 6.5",
  "npk":          {"n":40,"p":60,"k":50},
  "fertRec":      "1-2 sentence recommendation",
  "timeline":     [{"emoji":"🌰","label":"Sow","dur":"Week 1"}],
  "medicinalTags":["tag1","tag2"],
  "medicinalDesc":"Description",
  "hasMedicinal": true
}"""


def claude_enrich(model_result: dict, image_b64: str, mime: str) -> dict:
    """
    Mode 1: model already classified → ask Claude only for agronomic detail.
    We still pass the image so Claude can verify and fill in richer info.
    """
    client = anthropic.Anthropic(api_key=API_KEY)
    user_text = (
        f"The image has been classified by our CNN as: "
        f"'{model_result['crop_name']}' ({model_result['sci_name']}) "
        f"with {model_result['confidence']:.1f}% confidence.\n"
        "Using this classification (and the image as visual context), "
        "return the full JSON object described in the system prompt. "
        f"Use confidence = {round(model_result['confidence'])}."
    )
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_ENRICH,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",  "source": {"type": "base64", "media_type": mime, "data": image_b64}},
                {"type": "text",   "text": user_text},
            ]
        }]
    )
    raw  = "".join(b.text for b in response.content if b.type == "text")
    clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(clean)


def claude_vision(image_b64: str, mime: str) -> dict:
    """
    Mode 2: no model → let Claude do both identification and enrichment.
    """
    client = anthropic.Anthropic(api_key=API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_VISION,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": image_b64}},
                {"type": "text",  "text": "Identify this plant and return the full JSON object."},
            ]
        }]
    )
    raw   = "".join(b.text for b in response.content if b.type == "text")
    clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(clean)


# ─────────────────────────── ROUTES ───────────────────────────
@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts JSON: { "image": "<base64 string>", "mime": "image/jpeg" }
    Returns JSON matching the schema the frontend expects.
    """
    if not API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not set on server."}), 500

    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field in request body."}), 400

    image_b64 = data["image"]
    mime      = data.get("mime", "image/jpeg")

    # Decode bytes for model inference
    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"error": "Invalid base64 image data."}), 400

    try:
        if model is not None:
            # ── MODE 1: CNN → Claude enrichment ──────────────────
            log.info("Running CNN inference …")
            model_result = predict_with_model(image_bytes)
            log.info(f"  CNN says: {model_result['crop_name']} ({model_result['confidence']:.1f}%)")
            result = claude_enrich(model_result, image_b64, mime)
            result["_mode"] = "model+claude"
        else:
            # ── MODE 2: Pure Claude vision ────────────────────────
            log.info("Claude-only mode — running vision analysis …")
            result = claude_vision(image_b64, mime)
            result["_mode"] = "claude-only"

        return jsonify(result)

    except json.JSONDecodeError as e:
        log.error(f"Failed to parse Claude JSON: {e}")
        return jsonify({"error": "Claude returned invalid JSON. Please try again."}), 502
    except anthropic.APIError as e:
        log.error(f"Anthropic API error: {e}")
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":    "ok",
        "mode":      "model+claude" if model else "claude-only",
        "model_path": MODEL_PATH if model else None,
    })


# ─────────────────────────── ENTRY POINT ──────────────────────
if __name__ == "__main__":
    load_model()
    port = int(os.getenv("PORT", 5000))
    log.info(f"🌿 CropLens server starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)

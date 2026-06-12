# CropLens — Backend Setup

## Project structure

```
croplens/
├── app.py            ← Flask backend  (this file)
├── index.html        ← Frontend       (served by Flask at /)
├── requirements.txt
└── model.keras       ← Drop your trained model here (optional)
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."   # Linux/macOS
set ANTHROPIC_API_KEY=sk-ant-...        # Windows CMD

# 3. Run the server
python app.py

# 4. Open http://localhost:5000
```

## Modes

| Situation | Behaviour |
|-----------|-----------|
| `model.keras` / `model.h5` **not present** | Claude-only mode — Claude does full vision + identification |
| Model file **present** | Model+Claude mode — CNN classifies the crop, Claude fills agronomic details |

## Switching to your trained model

Once your Colab training finishes and you have the `.h5` / `.keras` file:

1. Copy it into the same folder as `app.py`.
2. Make sure the filename matches `MODEL_PATH` in `app.py` (default: `model.keras`).
3. Uncomment `tensorflow` in `requirements.txt` and run `pip install tensorflow`.
4. Restart the server — it will auto-detect and switch to Model+Claude mode.

You can also override the path via env var:
```bash
export CROPLENS_MODEL_PATH="/path/to/my_crop_model.h5"
```

## API

### `POST /analyze`
```json
// Request
{ "image": "<base64 string>", "mime": "image/jpeg" }

// Response  (matches frontend renderResult schema)
{
  "cropName": "Black Cumin",
  "sciName": "Nigella sativa",
  "emoji": "🌱",
  "confidence": 92,
  "growingSeason": "March – May",
  ...
  "_mode": "model+claude"   // or "claude-only"
}
```

### `GET /health`
```json
{ "status": "ok", "mode": "claude-only", "model_path": null }
```

## Class label order (must match your training)

Edit `CLASS_NAMES` in `app.py` if your Colab training used a different order:

```python
CLASS_NAMES = [
    "Black Cumin (Nigella sativa)",   # index 0
    "Sweet Pea (Lathyrus odoratus)",  # index 1
    "Allium Cepa (Onion)",            # index 2
]
```

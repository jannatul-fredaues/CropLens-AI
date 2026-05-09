# CropLens — AI Crop Identifier

A full-stack web app that uses your camera (or uploaded images) to identify crops and provide:
-  Crop name & scientific classification
-  Best planting time & seasons
-  Growing requirements (soil, water, sun, temp, fertilizer)
-  Medicinal properties & active compounds
-  Fun facts

---

##  Project Structure

```
crop-app/
├── backend/
│   ├── app.py           ← Flask API server
│   └── requirements.txt
└── frontend/
    └── index.html       ← Single-file frontend
```

---

##  Setup & Run

### 1. Backend (Python/Flask)

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."   # Mac/Linux
set ANTHROPIC_API_KEY=sk-ant-...        # Windows

# Start the server
python app.py
```

Server runs at: **http://localhost:5000**

---

### 2. Frontend

Simply open `frontend/index.html` in your browser.

> **Tip:** For camera access to work, browsers require either:
> - `localhost` (works out of the box), OR
> - HTTPS (use a local HTTPS server or deploy to a hosting provider)

---

##  How to Use

1. Click **Start Camera** to activate your device camera, then **Capture** to take a photo  
   — *or* —  
   Click the upload area to select an image from your device

2. Click **Analyze Crop**

3. View the full analysis including planting schedule, care guide, and medicinal info

---

##  Requirements

- Python 3.9+
- Anthropic API key (get one at https://console.anthropic.com)
- Modern browser with camera access (Chrome, Firefox, Safari)

---

##  Supported Crops

Works with virtually any plant or crop worldwide — vegetables, grains, herbs, fruits, medicinal plants, and more.


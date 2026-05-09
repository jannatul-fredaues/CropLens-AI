from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
import base64
import json
import re

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are an expert botanist and agricultural scientist with deep knowledge of crops, plants, and medicinal herbs worldwide. 

When given an image, analyze it and provide detailed information in the following JSON format ONLY (no extra text):

{
  "identified": true,
  "crop_name": "Common Name",
  "scientific_name": "Scientific name in italics placeholder",
  "confidence": "High/Medium/Low",
  "description": "Brief description of the crop",
  "planting_time": {
    "best_season": "Season name",
    "months": "Specific months",
    "climate_zones": "Suitable climate zones",
    "details": "Detailed planting time information"
  },
  "growing_requirements": {
    "soil": "Soil type and pH requirements",
    "water": "Watering needs",
    "sunlight": "Sunlight requirements",
    "temperature": "Temperature range",
    "fertilizer": "Fertilizer recommendations",
    "spacing": "Plant spacing",
    "harvest_time": "Time to harvest"
  },
  "medicinal_properties": {
    "has_medicinal_use": true,
    "primary_uses": ["use1", "use2", "use3"],
    "active_compounds": ["compound1", "compound2"],
    "traditional_medicine": "Traditional medicine uses",
    "modern_research": "Modern research findings",
    "caution": "Any warnings or contraindications"
  },
  "fun_fact": "An interesting fact about this crop"
}

If the image does NOT show a plant/crop, return:
{
  "identified": false,
  "message": "No plant or crop detected in the image. Please take a clear photo of a plant or crop."
}

Always respond with valid JSON only. No markdown, no extra text."""

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/analyze', methods=['POST'])
def analyze_crop():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = data['image']
        media_type = data.get('media_type', 'image/jpeg')

        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": "Please identify this crop/plant and provide all the requested information in JSON format."
                        }
                    ]
                }
            ]
        )

        result_text = response.content[0].text.strip()

        # Strip markdown code blocks if present
        result_text = re.sub(r'^```(?:json)?\s*', '', result_text)
        result_text = re.sub(r'\s*```$', '', result_text)
        result_text = result_text.strip()

        result = json.loads(result_text)
        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse AI response: {str(e)}"}), 500
    except anthropic.APIError as e:
        return jsonify({"error": f"AI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    print("🌿 Crop Identification Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

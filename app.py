from flask import Flask, request, jsonify
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# Load processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/")
def home():
    return "BLIP Caption API running!"

@app.route("/caption", methods=["POST"])
def caption():
    try:
        # Get JSON and extract image URL
        data = request.get_json()
        if not data or "image_url" not in data:
            return jsonify({"error": "Missing 'image_url' in request body"}), 400
        
        image_url = data["image_url"]

        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()  # Raise error for non-200 responses

        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Preprocess and generate caption
        inputs = processor(images=image, return_tensors="pt", padding=True)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return jsonify({"caption": caption})

    except requests.exceptions.RequestException as req_err:
        return jsonify({"error": f"Image download failed: {str(req_err)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

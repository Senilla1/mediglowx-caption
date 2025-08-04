from flask import Flask, request, jsonify
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/")
def home():
    return "BLIP Caption API running!"

@app.route("/caption", methods=["POST"])
def caption():
    try:
        # Parse JSON input
        data = request.get_json()
        if not data or "image_url" not in data:
            return jsonify({"error": "Missing 'image_url' in request body"}), 400

        image_url = data["image_url"]

        # Download image
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": f"Failed to download image, status code: {response.status_code}"}), 400

        # Open image
        try:
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

        # Preprocess image
        inputs = processor(images=image, return_tensors="pt", padding=True)

        # Generate caption
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return jsonify({"caption": caption})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

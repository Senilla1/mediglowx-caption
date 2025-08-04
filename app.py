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
        data = request.get_json()
        if not data or "image_url" not in data:
            return jsonify({"error": "Missing 'image_url' in request body"}), 400

        image_url = data["image_url"]
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400

        try:
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            return jsonify({"error": f"Failed to open image: {str(e)}"}), 400

        inputs = processor(images=image, return_tensors="pt", padding=True)
        if not inputs:
            return jsonify({"error": "Failed to preprocess image"}), 500

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return jsonify({"caption": caption})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

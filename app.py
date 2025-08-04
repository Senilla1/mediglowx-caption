from flask import Flask, request, jsonify
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from io import BytesIO

app = Flask(__name__)

# Load processor and model once when the server starts
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/caption", methods=["POST"])
def caption():
    try:
        # Parse JSON input
        data = request.get_json()
        image_url = data.get("image_url")

        if not image_url:
            return jsonify({"error": "Missing image_url"}), 400

        # Download image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')

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

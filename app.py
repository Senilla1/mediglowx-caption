from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/caption", methods=["POST"])
def generate_caption():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' URL in request"}), 400

    image_url = data["image"]

    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to load image from URL: {str(e)}"}), 400

    try:
        prompt = (
            "Describe the person's skin condition in detail, focusing on skin tone, texture, "
            "fine lines, wrinkles, dark spots, redness, irritation, acne, puffiness, under-eye bags, "
            "sagging skin, oily areas, and dry or flaky patches."
        )

        # ❗️ Kép + prompt együtt
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=100)
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]

        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": f"Failed to generate caption: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def home():
    return "Captioning service is running."

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

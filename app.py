from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
import traceback

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/caption", methods=["POST"])
def caption():
    try:
        data = request.get_json()
        image_url = data.get("image_url")

        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400

        print(f"[INFO] Received image URL: {image_url}")

        response = requests.get(image_url, stream=True)
        print(f"[INFO] Response status code: {response.status_code}")

        if response.status_code != 200:
            return jsonify({"error": f"Failed to fetch image, status code {response.status_code}"}), 400

        raw_image = Image.open(response.raw).convert("RGB")

        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        print(f"[INFO] Generated caption: {caption}")

        return jsonify({"caption": caption})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

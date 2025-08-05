from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import os

app = Flask(__name__)

# Use the instruction-following BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

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
        return jsonify({"error": f"Failed to load image from URL ({str(e)})"}), 400

    # 🧠 Prompt a konkrét bőrproblémákra
    prompt = "What skin problems are visible?"

    try:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=100)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": f"Failed to generate caption ({str(e)})"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

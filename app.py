from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, Blip2ForConditionalGeneration
import torch
import os

app = Flask(__name__)

# Modell és processzor betöltése (BLIP2 - instruction following)
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto", torch_dtype=torch.float16)

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

    # ÚJ, részletes prompt – bőranalízishez
    prompt = (
        "Please analyze the person's face in this photo and describe any visible skin concerns in detail, "
        "including wrinkles, fine lines, acne, redness, dark spots, uneven texture, or puffiness under the eyes."
    )

    try:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return jsonify({"caption": generated_text})

    except Exception as e:
        return jsonify({"error": f"Failed to generate caption ({str(e)})"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

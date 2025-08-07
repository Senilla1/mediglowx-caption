from flask import Flask, request, jsonify
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from io import BytesIO

app = Flask(__name__)

# 🔧 Modell betöltése – "Large" verzió
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/caption", methods=["POST"])
def caption_image():
    try:
        data = request.get_json()
        image_url = data.get("image")

        # 🟡 Debug – írd ki a kapott URL-t
        print("🔍 Kép URL:", image_url)

        if not image_url:
            return jsonify({"error": "No image_url provided"}), 400

        # 🔧 Töltsd le a képet
        response = requests.get(image_url)
        img_bytes = response.content

        # 💾 Debug – mentsd le a képet, amit a szerver ténylegesen lát
        with open("debug_kep.jpg", "wb") as f:
            f.write(img_bytes)

        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return jsonify({"caption": caption})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()

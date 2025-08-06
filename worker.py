from flask import Flask, request, jsonify
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from io import BytesIO

app = Flask(__name__)

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/caption', methods=['POST'])
def caption_image():
    try:
        data = request.get_json()
        image_url = data.get("image_url")

        if not image_url:
            return jsonify({"error": "No image_url provided"}), 400

        image = Image.open(BytesIO(requests.get(image_url).content)).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()

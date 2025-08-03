from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Hugging Face modell URL
HF_URL = "https://senilla-mediglowx-caption.hf.space/run/predict"

@app.route("/caption", methods=["POST"])
def caption():
    try:
        # Érkező JSON adatok
        data = request.get_json()

        # Kép URL kinyerése a JSON-ből
        image_url = data["data"][0] if "data" in data and isinstance(data["data"], list) else None
        if not image_url:
            return jsonify({"error": "Nincs érvényes kép URL"}), 400

        # API kérés összeállítása Hugging Face modellhez
        payload = {
            "data": [image_url]
        }

        # Küldés a Hugging Face modellnek
        response = requests.post(HF_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        caption = result.get("data", ["Nem sikerült képaláírást generálni."])[0]

        return jsonify({"caption": caption})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "MediGlowX Caption AI is live!"})

@app.route("/caption", methods=["POST"])
def caption():
    # Placeholder működéshez, amíg az AI-modul nincs készre kapcsolva
    return jsonify({"caption": "Ez egy teszt képaláírás a MediGlowX rendszerből."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

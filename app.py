from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from pathlib import Path

app = Flask(__name__)
CORS(app)

MODEL_PATH = Path(__file__).parent / "model.pkl"
VECT_PATH = Path(__file__).parent / "vectorizer.pkl"

# Lazy load to allow running app even before training (with helpful error)
model = None
vectorizer = None
if MODEL_PATH.exists() and VECT_PATH.exists():
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECT_PATH, "rb"))

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    global model, vectorizer
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500

    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    # Optional confidence via decision function if available
    conf = None
    if hasattr(model, "decision_function"):
        try:
            score = model.decision_function(vec)[0]
            # Map to a 0-1-ish confidence using logistic-like transform
            import math
            conf = 1 / (1 + math.exp(-abs(score)))
        except Exception:
            conf = None

    return jsonify({"prediction": pred, "confidence": conf})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
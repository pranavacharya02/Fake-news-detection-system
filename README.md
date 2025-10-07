# Fake News Detection System

A minimal full‑stack project that classifies news text as **FAKE** or **REAL**.

## Tech Stack
- **Backend:** Python, Flask, scikit-learn, TF‑IDF + PassiveAggressiveClassifier
- **Frontend:** HTML, CSS, Vanilla JS
- **Data:** Small sample CSV included for demo (`backend/data/news_sample.csv`)

---

## How to Run Locally

### 1) Backend
```bash
cd backend
python -m venv .venv
# Activate the venv:
#   Windows: .venv\Scripts\activate
#   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py    # trains and saves model.pkl & vectorizer.pkl
python app.py            # starts http://127.0.0.1:5000
```

### 2) Frontend
Open `frontend/index.html` in your browser (use VS Code Live Server for best results).

---

## API
- `GET /health` → `{ "status": "ok", "model_loaded": true|false }`
- `POST /predict` with JSON `{ "text": "..." }` → `{ "prediction": "FAKE|REAL", "confidence": 0-1 }`

---

## Improve the Model
This repo ships with a tiny demo dataset. For a more realistic project:
- Replace `backend/data/news_sample.csv` with a larger dataset (e.g., Kaggle Fake News).
- Consider using logistic regression, linear SVM, or fine-tuned transformer models.
- Add preprocessing (lowercasing, punctuation/URL removal), and experiment with n‑grams.

---

## Deploy
- **Backend:** Render/Fly/Heroku
- **Frontend:** GitHub Pages/Netlify/Vercel
- **Tip:** Add an environment variable for the API base URL and enable CORS as needed.

---

## Repo Setup
```bash
git init
git add .
git commit -m "Initial commit: Fake News Detection System"
git branch -M main
git remote add origin https://github.com/<your-username>/fake-news-detection.git
git push -u origin main
```

---

## License
MIT
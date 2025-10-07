import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from pathlib import Path

# Paths
DATA_PATH = Path(__file__).parent / "data" / "news_sample.csv"
MODEL_PATH = Path(__file__).parent / "model.pkl"
VECT_PATH = Path(__file__).parent / "vectorizer.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Ensure no NaNs
    df['text'] = df['text'].fillna("")
    df['label'] = df['label'].fillna("REAL")
    return df

def train():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = PassiveAggressiveClassifier(max_iter=100, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECT_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print("Saved:", MODEL_PATH.name, VECT_PATH.name)

if __name__ == "__main__":
    train()
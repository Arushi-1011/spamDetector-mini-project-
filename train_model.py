from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "spam.csv"
MODEL_PATH = ROOT / "spam_classifier_model.joblib"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[["label", "message"]].dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""].drop_duplicates()
    return df

def main():
    df = load_data()
    X = df["message"]
    y = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, stop_words="english"
        )),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ))
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "dataset_rows": int(len(df)),
        "ham_count": int((y == 0).sum()),
        "spam_count": int((y == 1).sum()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_spam": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_spam": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_spam": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
    }

    joblib.dump(model, MODEL_PATH)
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", classification_report(y_test, predictions, target_names=["Ham", "Spam"]))

if __name__ == "__main__":
    main()

"""
AI Healthcare Chatbot – Heart Disease Risk Prediction
======================================================
Mini Project | B.Tech CSE | Sem V – AIML
Dataset  : UCI Heart Disease Dataset (heart.csv)
Model    : Random Forest Classifier (scikit-learn)
Author   : Rahul Sharma, Priya Mehta
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "heart_model.pkl"

# ─── Feature metadata ─────────────────────────────────────────────────────────
QUESTIONS = [
    ("age",      "Enter your age (years, e.g. 52): ",          int,   (20, 100)),
    ("sex",      "Enter sex (1 = Male, 0 = Female): ",         int,   (0, 1)),
    ("cp",       "Chest pain type (0-None, 1-Typical Angina, 2-Atypical Angina, 3-Non-anginal, 4-Asymptomatic): ", int, (0, 4)),
    ("trestbps", "Resting blood pressure in mmHg (e.g. 130): ", int,  (80, 220)),
    ("chol",     "Serum cholesterol in mg/dl (e.g. 250): ",    int,   (100, 600)),
    ("fbs",      "Fasting blood sugar > 120 mg/dl? (1=Yes, 0=No): ", int, (0, 1)),
    ("restecg",  "Resting ECG result (0=Normal, 1=ST-T abnormality, 2=LV hypertrophy): ", int, (0, 2)),
    ("thalach",  "Maximum heart rate achieved (e.g. 150): ",   int,   (60, 210)),
    ("exang",    "Exercise-induced angina? (1=Yes, 0=No): ",   int,   (0, 1)),
    ("oldpeak",  "ST depression induced by exercise (e.g. 1.5): ", float, (0.0, 7.0)),
    ("slope",    "Slope of peak exercise ST segment (0=Upsloping, 1=Flat, 2=Downsloping): ", int, (0, 2)),
    ("ca",       "Number of major vessels coloured by fluoroscopy (0-3): ", int, (0, 3)),
    ("thal",     "Thalassaemia (3=Normal, 6=Fixed defect, 7=Reversible defect): ", int, [3, 6, 7]),
]

FEATURE_NAMES = [q[0] for q in QUESTIONS]

RISK_LABELS = {0: "LOW RISK – No significant heart disease detected.",
               1: "HIGH RISK – Significant risk of heart disease detected."}

ADVICE = {
    0: "Maintain a healthy diet, regular exercise, and routine check-ups.",
    1: "Please consult a cardiologist immediately for further evaluation."
}


# ─── Model utilities ──────────────────────────────────────────────────────────

def load_or_train_model():
    """Load pre-trained model or train a new one if not found."""
    if MODEL_PATH.exists():
        print("[INFO] Loading pre-trained model ...")
        return joblib.load(MODEL_PATH)
    print("[INFO] Model not found – training on heart.csv ...")
    return train_and_save_model()


def train_and_save_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    csv_path = Path(__file__).parent / "heart.csv"
    if not csv_path.exists():
        print("[ERROR] heart.csv not found. Please run  python download_data.py  first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Model trained  |  Test Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    joblib.dump(clf, MODEL_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")
    return clf


# ─── Chatbot dialogue ─────────────────────────────────────────────────────────

def greet():
    print("\n" + "=" * 60)
    print("   🏥  AI Healthcare Chatbot – Heart Disease Risk Predictor")
    print("=" * 60)
    print("Hello! I'm MedBot, your AI health assistant.")
    print("I will ask you a few clinical questions to assess your")
    print("risk of heart disease using a Machine Learning model.\n")
    print("⚠️  DISCLAIMER: This tool is for educational purposes only.")
    print("   It does NOT replace professional medical advice.\n")


def ask_question(label, prompt, dtype, valid):
    while True:
        try:
            raw = input(f"  → {prompt}").strip()
            value = dtype(raw)
            if isinstance(valid, list):
                if value not in valid:
                    raise ValueError
            else:
                lo, hi = valid
                if not (lo <= value <= hi):
                    raise ValueError
            return value
        except (ValueError, TypeError):
            if isinstance(valid, list):
                print(f"    [!] Please enter one of {valid}")
            else:
                print(f"    [!] Please enter a value between {valid[0]} and {valid[1]}")


def collect_patient_data():
    print("─" * 60)
    print("  Please answer the following questions:")
    print("─" * 60)
    data = {}
    for label, prompt, dtype, valid in QUESTIONS:
        data[label] = ask_question(label, prompt, dtype, valid)
    return data


def explain_prediction(model, patient_data):
    """Print top-3 contributing features for the prediction."""
    importances = model.feature_importances_
    fi = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    print("\n  📊 Top Contributing Factors:")
    for feat, score in fi[:3]:
        print(f"     • {feat:12s}  (importance: {score:.3f})  =  {patient_data[feat]}")


def predict_and_respond(model, patient_data):
    features = np.array([[patient_data[f] for f in FEATURE_NAMES]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][prediction]

    print("\n" + "=" * 60)
    print(f"  🔍 PREDICTION:  {RISK_LABELS[prediction]}")
    print(f"  📈 Confidence:  {proba:.1%}")
    print("=" * 60)
    explain_prediction(model, patient_data)
    print(f"\n  💡 Advice: {ADVICE[prediction]}")
    print("─" * 60)


def run_again():
    ans = input("\n  Would you like to assess another patient? (yes/no): ").strip().lower()
    return ans in ("yes", "y")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    model = load_or_train_model()
    greet()
    while True:
        patient_data = collect_patient_data()
        predict_and_respond(model, patient_data)
        if not run_again():
            print("\n  Thank you for using MedBot. Stay healthy! 🌿\n")
            break


if __name__ == "__main__":
    main()

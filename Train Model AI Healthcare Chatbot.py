"""
train_model.py
==============
Trains a Random Forest classifier on the UCI Heart Disease dataset,
evaluates performance, saves the model, and generates analysis charts.

Run: python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)

FEATURE_NAMES = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak","slope","ca","thal"
]

def load_data():
    csv_path = Path(__file__).parent / "heart.csv"
    if not csv_path.exists():
        print("[INFO] heart.csv not found – generating synthetic dataset ...")
        return generate_synthetic_data()
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} records from heart.csv")
    return df


def generate_synthetic_data(n=303, seed=42):
    """Reproduce a synthetic UCI-like dataset for demo / CI purposes."""
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 77, n)
    data = {
        "age":      age,
        "sex":      rng.integers(0, 2, n),
        "cp":       rng.integers(0, 4, n),
        "trestbps": rng.integers(94, 200, n),
        "chol":     rng.integers(126, 564, n),
        "fbs":      rng.integers(0, 2, n),
        "restecg":  rng.integers(0, 3, n),
        "thalach":  rng.integers(71, 202, n),
        "exang":    rng.integers(0, 2, n),
        "oldpeak":  np.round(rng.uniform(0, 6.2, n), 1),
        "slope":    rng.integers(0, 3, n),
        "ca":       rng.integers(0, 4, n),
        "thal":     rng.choice([3, 6, 7], n),
    }
    cp   = data["cp"]
    thal = data["thalach"]
    op   = data["oldpeak"]
    ca   = data["ca"]
    target = (((age > 55) & (cp > 1) & (thal < 140)) |
              ((op > 3)   & (ca > 1))).astype(int)
    data["target"] = target
    df = pd.DataFrame(data)
    df.to_csv(Path(__file__).parent / "heart.csv", index=False)
    print(f"[INFO] Synthetic dataset saved → heart.csv  ({n} records)")
    return df


def train(df):
    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cv   = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    print("\n" + "=" * 50)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.2%}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  5-Fold CV : {cv.mean():.2%} ± {cv.std():.2%}")
    print("─" * 50)
    print(classification_report(y_test, y_pred,
                                target_names=["No Disease", "Disease"]))

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = Path(__file__).parent / "heart_model.pkl"
    joblib.dump(clf, model_path)
    print(f"[INFO] Model saved → {model_path}\n")

    return clf, X_train, X_test, y_train, y_test, y_pred, y_proba


def plot_all(clf, X_test, y_test, y_pred, y_proba, df):
    charts_dir = Path(__file__).parent / "charts"
    charts_dir.mkdir(exist_ok=True)

    # 1. Feature Importance
    fi = pd.Series(clf.feature_importances_, index=FEATURE_NAMES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if v < fi.median() else "#F44336" for v in fi]
    fi.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Feature Importance – Random Forest", fontweight="bold")
    ax.set_xlabel("Importance Score")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(charts_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(charts_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#F44336", lw=2, label=f"ROC (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(charts_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Age Distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[df.target == 0]["age"], bins=15, alpha=0.7, color="#4CAF50", label="No Disease")
    ax.hist(df[df.target == 1]["age"], bins=15, alpha=0.7, color="#F44336", label="Disease")
    ax.set_xlabel("Age"); ax.set_ylabel("Count")
    ax.set_title("Age Distribution by Disease Status", fontweight="bold")
    ax.legend()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(charts_dir / "age_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Charts saved → {charts_dir}/")


if __name__ == "__main__":
    df = load_data()
    clf, X_train, X_test, y_train, y_test, y_pred, y_proba = train(df)
    plot_all(clf, X_test, y_test, y_pred, y_proba, df)
    print("[DONE] Training complete. Run  python chatbot.py  to launch the chatbot.")

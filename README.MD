# 🏥 AI Chatbot for Healthcare – Heart Disease Risk Predictor

> **Mini Project | B.Tech CSE – Semester V | AIML**

A command-line AI chatbot that predicts heart disease risk using a **Random Forest classifier** trained on the UCI Heart Disease dataset. The chatbot collects 13 clinical parameters through a guided conversation and returns a risk assessment with explainability.

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| Source | [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) |
| Records | 303 patient samples |
| Features | 13 clinical features + 1 binary target |
| Target | 0 = No Disease, 1 = Disease |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ai-healthcare-chatbot.git
cd ai-healthcare-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates heart_model.pkl + charts/)
python train_model.py

# 4. Launch the chatbot
python chatbot.py
```

---

## 📁 Project Structure

```
ai-healthcare-chatbot/
│
├── chatbot.py          # Main chatbot with dialogue manager
├── train_model.py      # Model training, evaluation & chart generation
├── heart.csv           # UCI Heart Disease dataset (auto-generated if missing)
├── heart_model.pkl     # Saved Random Forest model (after training)
├── requirements.txt    # Python dependencies
├── charts/             # Generated visualisation charts
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── age_distribution.png
└── README.md
```

---

## 🤖 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **93.44%** |
| Precision (Disease) | 100% |
| Recall (Disease) | 82% |
| F1-Score (Disease) | 90% |
| ROC-AUC | ~0.97 |

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **scikit-learn** – Random Forest Classifier
- **pandas / NumPy** – Data processing
- **Matplotlib** – Visualisation
- **joblib** – Model serialisation

---

## 💬 Sample Chatbot Session

```
============================================================
   🏥  AI Healthcare Chatbot – Heart Disease Risk Predictor
============================================================
Hello! I'm MedBot, your AI health assistant.

  → Enter your age (years, e.g. 52): 61
  → Enter sex (1 = Male, 0 = Female): 1
  → Chest pain type (0-4): 3
  ...

============================================================
  🔍 PREDICTION:  HIGH RISK – Significant risk detected.
  📈 Confidence:  87.0%
============================================================
  📊 Top Contributing Factors:
     • thalach      (importance: 0.142)  =  128
     • oldpeak      (importance: 0.131)  =  3.5
     • ca           (importance: 0.128)  =  2

  💡 Advice: Please consult a cardiologist immediately.
```

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. It does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.

---

## 📌 Mapped SDGs

- **SDG 3** – Good Health and Well-Being
- **SDG 10** – Reduced Inequalities

---

## 👥 Authors

- Sehaj Preet (70314803123)
- FaiZ ParveZ (70114803123)

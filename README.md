# Sentiment Analysis using Machine Learning (Python)

This project performs **sentiment analysis** using Python and Machine Learning (Logistic Regression + TF-IDF)**.

## 🚀 Features
- Clean ML Pipeline (scikit-learn)
- Trainable model using labeled text data
- Predict sentiment for new input
- Saved model export (`joblib`)
- Minimal dataset included

## 📂 Project Structure
```
sentiment-analysis-ml/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── sentiment_data.csv
│
├── models/
│   └── sentiment_model.joblib (generated after training)
│
└── src/
    ├── train_model.py
    └── predict.py
```

## 🔧 Installation
```bash
pip install -r requirements.txt
```

## 🏋️ Train Model
```bash
python src/train_model.py
```

## 🔍 Predict Sentiment
```bash
python src/predict.py "This product is great!"
```

## 📊 Sample Output
```
Input text: This product is great!
Predicted label: positive
Probability (positive): 0.92
Probability (negative): 0.08
```

---

### 📌 Dataset Format
CSV file must have columns:
```
text,label
```

---

### 📜 License
This project is open-source and free to use.

import sys
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"

model = joblib.load(MODEL_PATH)

text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter text: ")

pred = model.predict([text])[0]
prob = model.predict_proba([text])[0]

print("Text:", text)
print("Predicted:", pred)
print("Probabilities:", dict(zip(model.classes_, prob)))

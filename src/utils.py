import joblib
from pathlib import Path

MODEL_PATH = Path("artifacts/linear_model.joblib")

def load_model(path=MODEL_PATH):
    return joblib.load(path)

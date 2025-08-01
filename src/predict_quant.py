from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

ARTIFACT_DIR = Path("artifacts")
Q_MODEL = ARTIFACT_DIR / "quant_model.joblib"

def load_quantised_model():
    qm = joblib.load(Q_MODEL)
    q_params = qm["q_params"]
    mins = qm["mins"]
    scales = qm["scales"]
    weights = q_params.astype(np.float32) * scales + mins
    coefs = weights[:-1]
    intercept = weights[-1]
    return coefs, intercept

def main():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    coefs, intercept = load_quantised_model()
    preds = np.dot(X_test.to_numpy(), coefs) + intercept
    print("Quantised model predictions (first 5):", np.round(preds[:5], 3))
    print("Quantised coefficients:", np.round(coefs, 4))
    print("Quantised intercept:", np.round(intercept, 4))
    y_true = y.iloc[X_test.index]
    print("Quantised R²:", np.round(r2_score(y_true, preds), 4))
    print("Quantised RMSE:", np.round(mean_squared_error(y_true, preds, squared=False), 4))

if __name__ == "__main__":
    main()

"""
Quantise scikit-learn LinearRegression coefficients to uint8 and
store both raw and quantised versions.
For demo purposes we use simple min‑max scaling.
"""
from pathlib import Path
import joblib
import numpy as np
from src.utils import load_model

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

RAW_PATH = ARTIFACT_DIR / "unquant_params.joblib"
Q_PATH   = ARTIFACT_DIR / "quant_params.joblib"
SCALE_PATH = ARTIFACT_DIR / "scale.joblib"  # for de‑quantisation

def quantise():
    model = load_model()
    coefs = model.coef_.astype(np.float32)
    intercept = np.array([model.intercept_], dtype=np.float32)

    params = np.concatenate([coefs, intercept])

    # Min‑max → uint8
    p_min, p_max = params.min(), params.max()
    scale = (p_max - p_min) / 255.0
    q_params = np.round((params - p_min) / scale).astype(np.uint8)

    joblib.dump({"coefs": coefs, "intercept": intercept}, RAW_PATH)
    joblib.dump(q_params, Q_PATH)
    joblib.dump({"min": p_min, "scale": scale}, SCALE_PATH)

    print("✔ Quantised parameters saved")
    return q_params, p_min, scale

def dequantise(q_params, p_min, scale):
    return q_params.astype(np.float32) * scale + p_min

if __name__ == "__main__":
    quantise()

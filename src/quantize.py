from pathlib import Path
import joblib
import numpy as np
from src.utils import load_model

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

def quantise():
    model = load_model()
    coefs = model.coef_.astype(np.float32)
    intercept = np.array([model.intercept_], dtype=np.float32)
    params = np.concatenate([coefs, intercept])

    # Per-coefficient quantisation: range ±0.05 around each value
    mins = params - 0.05
    maxs = params + 0.05
    scales = (maxs - mins) / 255.0
    q_params = np.round((params - mins) / scales).astype(np.uint8)

    joblib.dump({
        "q_params": q_params,
        "mins": mins,
        "scales": scales
    }, ARTIFACT_DIR / "quant_model.joblib")

    print("✔ Quantised parameters saved (per-coefficient)")
    print("Original coefficients:", np.round(coefs, 4))
    print("Original intercept:", np.round(intercept[0], 4))

if __name__ == "__main__":
    quantise()

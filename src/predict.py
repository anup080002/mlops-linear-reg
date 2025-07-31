"""
Load (quantised or original) model, run one batch of predictions,
and print example outputs.
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
from src.utils import load_model


def main():
    data = fetch_california_housing(as_frame=True)

    # >>> use only the 8 feature columns, NOT the target
    X = data.data
    y = data.target
    # <<<

    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = load_model()
    preds = model.predict(X_test.iloc[:5])
    print("Sample predictions:", np.round(preds, 3))


if __name__ == "__main__":
    main()

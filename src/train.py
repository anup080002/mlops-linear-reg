"""
Train a Linear Regression model on the California Housing dataset,
report metrics, and persist the model to disk.
"""
from pathlib import Path
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

MODEL_DIR = Path("artifacts")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
MODEL_PATH = MODEL_DIR / "linear_model.joblib"


def load_data():
    """
    Returns
    -------
    X : pandas.DataFrame  (8 feature columns only)
    y : pandas.Series     (MedHouseVal target)
    """
    data = fetch_california_housing(as_frame=True)
    X = data.data        # <-- *only* features; target column excluded
    y = data.target
    return X, y


def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"✔ R²: {r2:.4f} | RMSE: {rmse:.4f}")
    joblib.dump(lr, MODEL_PATH)
    return lr, r2


if __name__ == "__main__":
    train()

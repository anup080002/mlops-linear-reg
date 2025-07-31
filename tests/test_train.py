import pytest
from src import train

@pytest.fixture(scope="module")
def model_and_score():
    model, r2 = train.train()
    return model, r2

def test_model_instance(model_and_score):
    model, _ = model_and_score
    from sklearn.linear_model import LinearRegression
    assert isinstance(model, LinearRegression)

def test_coefficients_exist(model_and_score):
    model, _ = model_and_score
    assert hasattr(model, "coef_"), "Model not trained"

def test_r2_threshold(model_and_score):
    _, r2 = model_and_score
    assert r2 > 0.5, f"R² too low: {r2}"

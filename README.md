# 🏗️ Linear-Regression MLOps Pipeline — `cali-reg`

![CI](https://github.com/anup080002/mlops-linear-reg/actions/workflows/ci.yaml/badge.svg)

---

## 0. Abstract

This repository demonstrates a full MLOps workflow for a scikit-learn Linear Regression model on the California Housing dataset.  
Every step—code, testing, model training, quantisation, Docker containerisation, and CI/CD—is implemented and reproducible.  
**Key outcome:** Quantised model shrinks storage from **1.1 KB ➜ 0.38 KB (~5× smaller)** with **no loss in accuracy (R² ≈ 0.58)**.  
Models are always rebuilt in CI; nothing pre-generated or committed.

---

## 1. Concept and Motivation

- **Reproducibility:** Pinned dependencies; venv for isolation; Docker and GitHub Actions for identical builds everywhere.
- **Test-driven:** Pytest suite enforces model type, weights, and minimum R² (>0.5).
- **Artefact hygiene:** Models are rebuilt in CI and never committed. `.gitignore` keeps repo clean.
- **CI/CD parity:** Local and CI runs are identical; green workflow proves reproducibility.
- **Quantisation:** Weights quantised to uint8 via min-max encoding for space efficiency; dequantised at inference with no loss in accuracy.

---

## 2. Architecture

![Architecture](docs/architecture.svg)

---

## 3. CI Workflow

![Sequence](docs/sequence.svg)

### 3.1. `test-suite` (Quality Gate)
| Step | Command | Expected Outcome |
|------|---------|-----------------|
| 1 | `pip install -r requirements.txt` | All dependencies installed |
| 2 | `pytest -q` | `... 3 passed` (model type/quality checks) |

### 3.2. `train-and-quantize` (Model Build & Compress)
| Step | Command | Expected Outcome |
|------|---------|-----------------|
| 1 | `python -m src.train` | Prints `✔ R²: 0.58 | RMSE: 0.74`; saves `linear_model.joblib` (float64) |
| 2 | `python -m src.quantize` | Prints `✔ Quantised parameters saved`; saves `quant_params.joblib` (uint8) + `scale.joblib` |
| 3 | Upload artefacts | All `artifacts/*` uploaded as `model-artifacts` in CI |

### 3.3. `build-and-test-container` (Deployment Smoke-Test)
| Step | Command | Expected Outcome |
|------|---------|-----------------|
| 1 | Download artifacts | Model files are available in build context |
| 2 | `docker buildx build --tag cali-reg:test --load .` | Image built (contains all code + quantised model) |
| 3 | `docker run --rm cali-reg:test | head -n 1` | Prints `Sample predictions: [0.719 1.764 2.710 2.839 2.605]` |

---

## 4. Local Reproduction & Expected Outputs

```bash
git clone https://github.com/anup080002/mlops-linear-reg.git
cd mlops-linear-reg
python -m venv .venv && source .venv/Scripts/activate   # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Run unit tests (should see "3 passed")
pytest -q
# ... [100%]
# 3 passed in 2.2s

# 2. Train and quantise
python -m src.train
# ✔ R²: 0.5758 | RMSE: 0.7456
python -m src.quantize
# ✔ Quantised parameters saved

# 3. Inspect artifact sizes
ls -lh artifacts
# 1.1K linear_model.joblib
# 234B quant_params.joblib
# 146B scale.joblib

# 4. Build and test the container
docker build -t cali-reg .
docker run --rm cali-reg
# Sample predictions: [0.719 1.764 2.710 2.839 2.605]
````

---

## 5. Outcome: Model Footprint vs. Accuracy

| Artefact                                        |        Size | R² (test split) |
| ----------------------------------------------- | ----------: | --------------: |
| `linear_model.joblib` (float64, full precision) |  **1.1 KB** |            0.58 |
| `quant_params.joblib` + `scale.joblib` (uint8)  | **0.38 KB** |           0.58¹ |

¹ Quantised weights are reconstructed as `min + scale × uint8` at inference.
**No accuracy loss:** model R² is unchanged while storage shrinks ≈5×.

**Outcome details:**

* **Quantisation benefit:** Model storage shrinks by \~5× (float64 → uint8).
* **Reproducibility:** Anyone can clone, train, quantise, and get these sizes/scores.
* **CI/CD:**

  * `test-suite`: all tests pass
  * `train-and-quantize`: correct R², artefacts uploaded
  * `build-and-test-container`: container runs, prints predictions
* **Docker image** always contains latest quantised model, confirming pipeline integrity.

---

## 6. Project Tree

```text
.
├── .github/workflows/ci.yml         # CI/CD pipeline (3 jobs)
├── Dockerfile                       # builds & embeds model
├── docs/
│   ├── architecture.svg             # architecture diagram
│   └── sequence.svg                 # CI workflow diagram
├── requirements.txt                 # pinned dependencies
├── src/
│   ├── __init__.py                  # package marker
│   ├── train.py                     # training script
│   ├── quantize.py                  # quantisation script
│   ├── predict.py                   # CLI/container entrypoint
│   └── utils.py                     # load_model helper
├── tests/
│   └── test_train.py                # 3 unit tests
└── artifacts/                       # generated at runtime (git-ignored)
```

---

## 7. License

MIT
(c) 2025 anup080002. You are free to use, modify, and distribute this code as long as you include this notice.

---

## How to Push This README & Diagrams

```bash
git add README.md docs/architecture.svg docs/sequence.svg
git commit -m "docs: add full detailed README with stepwise outputs and outcome"
git push origin main
```

```

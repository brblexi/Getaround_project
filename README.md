# GetAround — delay analysis and price prediction

End-to-end machine learning project on car-sharing data: an interactive dashboard
that quantifies a product decision, a pricing model tracked with MLflow, and a
containerised API that serves it. Both applications are deployed and live.

| | Live | Source |
|---|---|---|
| **Dashboard** — should GetAround enforce a minimum delay between rentals? | [alexbarbier-getaround-dashboard.hf.space](https://alexbarbier-getaround-dashboard.hf.space) | [`dashboard/`](dashboard) |
| **Prediction API** — daily rental price from a car's characteristics | [alexbarbier-getaround-api.hf.space/docs](https://alexbarbier-getaround-api.hf.space/docs) | [`api/`](api) |

---

## The problem

GetAround is a car-sharing marketplace. Two questions, one dataset each.

**1. Late returns.** When a driver returns a car late, the next rental of the same
car is disrupted. Product wants to enforce a minimum delay between two rentals —
but every blocked booking is lost revenue. How wide should the buffer be, and
should it apply to all cars or only to Connect ones?

**2. Pricing.** Owners need a suggested daily price when they list a car.

## What the analysis found

On 21,310 rentals:

- **57%** of returns are late, with a median lateness of ~53 minutes.
- Only **9%** of rentals are back-to-back, which caps what any delay rule can achieve.
- Within that subset, **12.6%** are actually impacted — the previous car came back
  after the next rental's planned start.
- Impact concentrates on **mobile check-ins (15.9%) versus Connect (8.7%)**.
- Impacted rentals are cancelled **17%** of the time against **11%** otherwise.
- A **60–120 minute** threshold on *all* cars resolves 47–67% of problem hand-overs
  while blocking 1.9–3.1% of rentals. Recommended as an A/B test, not as a rollout:
  the 11%→17% cancellation gap is observational, and blocked bookings are a real cost.

The dashboard lets a product manager move the threshold and the scope and read the
trade-off directly.

## The model

Daily price regression on 4,840 cars, 13 features.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random forest (`n=100`, `max_depth=12`) | **16.99 €** | **10.97 €** | **0.748** |
| Linear regression (baseline) | 18.39 € | 12.40 € | 0.705 |

A typical prediction lands within ~11 € of the listed price, on prices averaging
121 €. The forest is served; the linear model is kept as the reference it has to beat, and both are compared against a predict-the-median baseline.

Two decisions worth surfacing:

- **A log transform of the target was tested and rejected.** The distribution is
  close to symmetric (skewness 0.61, mean 121.2 against median 119.0), and the log
  version scored worse on every metric.
- **`max_depth=12` is a deployment decision as much as a statistical one.** A depth
  sweep showed test RMSE flattening well before the depth at which the artifact grows
  from 2.3 MB to 62 MB. The sweep is reproduced from the tracking database in
  [`model/03_tracking_infrastructure.ipynb`](model/03_tracking_infrastructure.ipynb).

## Architecture

```
     training                       tracking                    serving
 ┌──────────────┐            ┌─────────────────────┐      ┌──────────────────┐
 │  train.py    │─ metrics ─▶│ PostgreSQL (Neon)   │      │ FastAPI + uvicorn│
 │  sklearn     │─ artifacts▶│ S3 bucket           │      │ Docker image     │
 │  Pipeline    │            └─────────────────────┘      │  └ model.joblib  │
 └──────┬───────┘                                          └────────▲─────────┘
        │                                                           │
        └──────────────── joblib.dump ──────────────────────────────┘
```

The pipeline is serialised twice, independently: once by MLflow for traceability,
once by `joblib` for deployment. Nothing links the two calls, so the API has no
runtime dependency on MLflow, Neon or S3 — if the tracking stack went down, it would
keep serving. The trade-off is that updating the served model means rebuilding the
image. A model registry would close that gap and is the natural next step.

The **whole pipeline** is serialised, not just the estimator, so the API receives raw
features and preprocessing cannot drift between training and serving.

## Repository

```
.
├── notebooks/
│   └── 01_data_exploration.ipynb          both datasets, EDA
├── dashboard/                             Part 1 — Streamlit app + Dockerfile
│                                          app.py renders, utils.py computes
├── model/                                 Part 2 — training and tracking
│   ├── 02_price_modelling.ipynb           cleaning, pipeline, training, MLflow
│   ├── 03_tracking_infrastructure.ipynb   reads the tracking backend in SQL
│   └── train.py                           the same run, in one command
└── api/                                   Part 3 — FastAPI + model.joblib
```

Notebooks 02 and 03 live in `model/` rather than in `notebooks/` because their
relative paths point there: `02` reads `data/` and writes `artifacts/`, `03` loads
the `.env` sitting beside it.

Each deployable component has its own `requirements.txt` and `Dockerfile`, so images
stay minimal. Each also has its own README covering local runs, Docker and
deployment.

## Running it

```bash
# Dashboard — http://localhost:8501
cd dashboard
pip install -r requirements.txt
streamlit run app.py

# Training (writes to mlruns/ locally, or to Neon + S3 if .env is populated)
cd model
pip install -r requirements.txt
python train.py

# API — http://localhost:8000/docs
cd api
pip install -r requirements.txt
uvicorn app:app --reload
```

The same commands work under PowerShell. Either application can also run as a
container:

```bash
docker build -t getaround-api ./api
docker run -p 7860:7860 getaround-api
```

## Tests

40 tests across three components, each run from its own folder:

| Folder | Tests | Covers |
|---|---|---|
| `dashboard/` | 23 | business rules on a hand-built fixture, plus a smoke test that renders the app |
| `api/` | 9 | response contract, schema validation, unknown-category rejection |
| `model/` | 8 | cleaning, feature groups, preprocessor, metrics |

CI runs the three suites, `ruff`, and both Docker builds on every push.

## Using the API

```bash
curl -X POST "https://alexbarbier-getaround-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"model_key":"Citroën","mileage":140000,"engine_power":100,"fuel":"diesel","paint_color":"black","car_type":"sedan","private_parking_available":true,"has_gps":true,"has_air_conditioning":false,"automatic_car":false,"has_getaround_connect":true,"has_speed_regulator":true,"winter_tires":false}'
```

```json
{"rental_price_per_day": 115.6}
```

```python
import requests

car = {
    "model_key": "Citroën", "mileage": 140000, "engine_power": 100,
    "fuel": "diesel", "paint_color": "black", "car_type": "sedan",
    "private_parking_available": True, "has_gps": True,
    "has_air_conditioning": False, "automatic_car": False,
    "has_getaround_connect": True, "has_speed_regulator": True,
    "winter_tires": False,
}
print(requests.post("https://alexbarbier-getaround-api.hf.space/predict", json=car).json())
```

Unknown categorical values are rejected with a `422` listing the accepted values,
rather than scored silently. `handle_unknown="ignore"` keeps the encoder from raising
on an unseen category, but it encodes it as all-zeros — so `"Citroen"` without the
diaeresis returned a confident 123.92 € instead of 115.60 €, with a 200 and no
warning. The API now reads the accepted values back from the fitted encoder and
rejects anything outside them. See [`api/README.md`](api/README.md).

## Limitations

- The cancellation gap between impacted and unimpacted rentals is observational.
  Impacted rentals differ in other ways (check-in type above all), so the threshold
  recommendation is a hypothesis to test, not a measured effect — and the gap itself
  has not been tested for significance.
- The delay dataset contains implausible values (−22,433 to +71,084 minutes) that are
  kept rather than corrected; they are clipped for display only.
- Prediction errors are not uniform: the model regresses towards the middle of the
  price distribution and is least reliable on premium listings, which is where a
  pricing suggestion matters most.
- The pricing model has no temporal or geographic features — seasonality and location
  are likely to matter and are absent from the dataset.

## Next steps

- MLflow model registry, so the API requests the Production version instead of
  embedding a frozen file.
- Permutation importance instead of impurity-based importance, which is biased towards
  high-cardinality features such as `model_key`.
- Monitoring on the served model: input distribution drift and prediction distribution.

---

*Stack: Python, pandas, scikit-learn, MLflow, Streamlit, Plotly, FastAPI, Docker,
PostgreSQL, S3, Hugging Face Spaces.*

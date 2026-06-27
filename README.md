# GetAround — Bloc 5 (Jedha CDSD)

End-to-end project for the GetAround case study: analysing the impact of late
car returns and serving a rental-price prediction model.

> **Status** — All three deliverables are complete: delay-analysis dashboard,
> pricing model with MLflow tracking, and the documented `/predict` API.

## Deliverables

| # | Deliverable | Stack | Status |
|---|-------------|-------|--------|
| 0 | **Exploration des données** (EDA) | Jupyter + pandas + seaborn | ✅ done |
| 1 | Delay-analysis **dashboard** | Streamlit + Plotly, Docker → HF Spaces | ✅ done |
| 2 | **Pricing model** + experiment tracking | scikit-learn + MLflow | ✅ done |
| 3 | **`/predict` API** (documented) | FastAPI + Docker → HF Spaces | ✅ done |

## Repository layout

```
getaround/
├── notebooks/            # Exploratory data analysis (EDA)
│   ├── 01_exploration_donnees.ipynb
│   └── requirements.txt
├── model/                # Part 2 — Pricing model + MLflow
│   ├── 02_modelisation_pricing.ipynb
│   ├── train.py
│   ├── data/  ·  artifacts/model.joblib  ·  requirements.txt
│   └── README.md
├── api/                  # Part 3 — FastAPI /predict endpoint
│   ├── app.py  ·  model.joblib  ·  Dockerfile  ·  requirements.txt
│   └── README.md
├── dashboard/            # Part 1 — Streamlit delay-analysis dashboard
│   ├── app.py            #   UI
│   ├── utils.py          #   data loading + analysis/simulation logic
│   ├── data/             #   source datasets
│   ├── Dockerfile        #   HF Spaces (Docker SDK), port 7860
│   ├── requirements.txt
│   ├── .streamlit/config.toml
│   └── README.md         #   Space README (YAML frontmatter)
├── .gitignore
└── README.md             # this file
```

## Part 1 — Delay analysis

Quantifies how often cars are returned late, how that impacts the *next*
driver (cancellations rise from ~11% to ~17% when impacted), and provides an
interactive simulator for the **minimum-delay-between-rentals** threshold and
scope (all cars vs Connect-only), showing the trade-off between problems
solved and rentals blocked.

See [`dashboard/README.md`](dashboard/README.md) to run or deploy.

## Links

- **Dashboard (HF Space):** _add URL after deployment_
- **API (HF Space):** _add URL after deployment_
- **GitHub repo:** _add URL_

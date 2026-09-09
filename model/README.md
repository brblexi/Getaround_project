# Part 2 — Rental price prediction model

Predicts `rental_price_per_day` from a car's characteristics. Supervised
regression, with experiment tracking in **MLflow**.

## Contents

```
model/
├── 02_price_modelling.ipynb          # narrative notebook (the full reasoning)
├── 03_tracking_infrastructure.ipynb  # reads the tracking backend in SQL
├── train.py                          # the same run, in one command
├── test_train.py                     # 8 tests on cleaning, pipeline, metrics
├── data/                             # pricing dataset
├── artifacts/model.joblib            # trained pipeline (consumed by the API)
├── artifacts/model.json              # what that artifact is (see below)
├── .env.example                      # configuration to copy to .env
└── requirements.txt
```

Both notebooks live here rather than in `notebooks/` because their relative paths
point at this folder: `02` reads `data/` and writes `artifacts/`, `03` loads the
`.env` sitting next to it. Run them from `model/`.

## Approach

Cleaning (3 implausible rows removed) → scikit-learn pipeline (`StandardScaler` +
`OneHotEncoder` + passthrough booleans) → `LinearRegression` (baseline) against
`RandomForestRegressor` → MLflow tracking → serialisation of the best model.

**Test-set results:** the random forest wins with RMSE ≈ 17 €, MAE ≈ 11 €,
R² ≈ 0.75 (linear baseline: RMSE ≈ 18.4 €, R² ≈ 0.70). Both are well ahead of the
predict-the-median reference, which `train.py` reports on every run.

## Installing the dependencies

Once, in the project's virtual environment:

```bash
pip install -r requirements.txt
```

## Running the training

`python train.py` is enough — every option has a default. The lines below are
**variants**, not a sequence: run one, and the options combine.

```bash
python train.py                                       # both models, defaults
python train.py --model rf                            # one model (linear | rf | both)
python train.py --n-estimators 300 --max-depth 20     # other forest hyperparameters
python train.py --experiment getaround-pricing-sweep  # a fresh MLflow experiment
```

The last one is how the depth sweep behind `max_depth=12` was produced, and it is
also the only way to send an existing project's runs to S3 — see the
`artifact_location` warning below.

`python train.py --help` lists the remaining options (`--test-size`, `--cv`,
`--output`, `--max-depth -1` for unlimited depth). That listing is generated from
the code, so unlike this README it cannot fall out of date.

Passing options rather than editing the code keeps every run reproducible: MLflow
records the parameters that were actually used, alongside the git commit and the
scikit-learn version.

### What a run produces

Beyond the MLflow run, two files land in `artifacts/`:

- **`model.joblib`** — the complete pipeline, compressed (~2.3 MB). This is the
  file the API serves; copy it to `api/` and rebuild the image to deploy it.
- **`model.json`** — a sidecar describing that artifact: model type, test metrics,
  timestamp, scikit-learn version, git commit, row count, seed. A bare `.joblib`
  says nothing about itself, and the scikit-learn version is what decides whether
  it will reload at all.

Cross-validation runs by default (`--cv 5`) and logs `cv_rmse_mean` and
`cv_rmse_std` alongside the holdout metrics — a single split with a fixed seed
gives one number with no sense of its variance.

## Experiment tracking: where the data lands

Tracking **switches automatically** on environment variables
(`get_tracking_uri()` + `setup_experiment()`), with a **backend store / artifact
store** split:

| Data | Goes to | Variable |
|---|---|---|
| Metrics, parameters, run metadata | **PostgreSQL / Neon** | `MLFLOW_TRACKING_URI` |
| Serialised model, logged files | **S3** | `MLFLOW_ARTIFACT_LOCATION` (+ AWS keys) |

`train.py` loads `.env` at startup (`load_dotenv`), so **nothing has to be exported
by hand** before a run. If the `.env` is missing — or the variables are not filled
in — the script falls back to **local** storage (`mlruns/`).

> **Why `load_dotenv` rather than reading a config file?** The code only ever
> reads `os.environ`, never the `.env` directly. That is the production contract:
> on a platform (Docker, a Hugging Face Space, ECS) the variables are **injected by
> the host** and no `.env` exists — the call finds nothing and does nothing. The
> `.env` is a local development convenience, with `override=False` so a variable
> already set in the environment always wins.

### Infrastructure prerequisites

To provision once, before the first remote run:

1. **A dedicated PostgreSQL database** (Neon) — do not reuse one that already holds
   MLflow tables from another project, or the experiments and registered models
   would mix. Use the **direct** endpoint (without `-pooler`): MLflow creates its
   schema through Alembic migrations on first connection, which a transaction-mode
   pooler handles poorly.
2. **An S3 bucket** — public access blocked, versioning off (MLflow does the
   versioning, one folder per run). Its region is **fixed at creation** and must
   match `AWS_DEFAULT_REGION` exactly.
3. **A dedicated IAM user** with a policy restricted to that single bucket
   (`ListBucket` + `GetBucketLocation` on the bucket, `PutObject` / `GetObject` /
   `DeleteObject` on `bucket/*`) rather than `AmazonS3FullAccess` — the keys live
   in a local file, so the blast radius of a leak is worth limiting.

### Configuration

Copy `.env.example` to `.env` and fill it in. Then:

```bash
python train.py    # metrics -> Neon, artifacts -> S3
```

Two confirmation lines appear at startup:

```
MLflow tracking -> remote backend
Artifacts -> s3://<bucket>/mlflow-artifacts
```

The second only appears **when the experiment is created**. Its absence means the
experiment already existed — see the warning below.

### Viewing the experiments

**Locally** (no variables set):

```bash
mlflow ui      # http://localhost:5000
```

Identical under PowerShell — but a local file store needs an opt-in first:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui
```

**Against Neon.** `mlflow ui` is a **separate process** and does not read the
`.env`: the URI has to be passed explicitly.

```powershell
# PowerShell
$env:MLFLOW_TRACKING_URI = ((Get-Content .env | Select-String "^MLFLOW_TRACKING_URI=") -split "=", 2)[1]
mlflow ui --backend-store-uri $env:MLFLOW_TRACKING_URI
```

```bash
# bash / Linux
export $(grep -v '^#' .env | xargs)
mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"
```

To browse a run's *Artifacts* tab as well, the server needs the AWS credentials
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) in its own
environment. Two stores, two authentications: Neon is enough for metrics and
parameters, AWS is required to download a model.

> ⚠️ **If the runs do not show up.** Three causes, in order. The database name at
> the end of the URI decides which backend the UI reads — point it at the wrong one
> and you get an interface that looks fine but lists someone else's experiments.
> The `&` in a Neon connection string terminates the command under PowerShell; run
> it from Git Bash or escape it as `%26`. And the search bar keeps filters between
> sessions: the metric is `test_rmse`, not `rmse`, and `params.model` is `"rf"` or
> `"linear"`.

> ⚠️ **Trap: `artifact_location` is frozen when the experiment is created.**
>
> The artifact location is a **column of the `experiments` table**, written once at
> `INSERT` and never recomputed. Setting `MLFLOW_ARTIFACT_LOCATION` **after** an
> experiment exists therefore has no effect on it — **and raises no error**.
>
> It follows that the environment variable does not apply by itself: it has to be
> **passed** to `create_experiment(name, artifact_location=…)`, which is what
> `setup_experiment()` does. The `Default` experiment, created internally by MLflow
> without that argument, is the control case: it keeps a local path even when the
> S3 variable is set.
>
> Check before training:
> ```python
> print(mlflow.get_experiment_by_name("getaround-pricing").artifact_location)
> ```
> If the result starts with `file:///`, artifacts are staying local. Delete the
> experiment, or use a fresh name (`python train.py --experiment getaround-pricing-s3`).

> 🔐 **Secrets.** The Neon URL (which contains a password) and the AWS keys live in
> `.env`, unversioned (see `.gitignore`) — never hard-coded, never on GitHub. They
> concern training only: the deployed API does no MLflow tracking, it loads a
> `.joblib` and predicts.

> 📦 **Why two stores?** A SQL database (Neon) is built for structured, queryable
> data — numbers and text — not for binary files; S3 is built for files. Neon keeps,
> for each run, the S3 URI of its artifacts (the `artifact_uri` column of the `runs`
> table), which is what links the two.

> The pipeline is serialised **twice, independently**, by two calls that nothing in
> `train.py` connects: `mlflow.sklearn.log_model()` sends one copy to **S3**, and
> `joblib.dump()` writes `artifacts/model.joblib` to disk. Two distinct purposes:
> the S3 artifact serves **traceability** (it ships with `MLmodel`,
> `requirements.txt`, `python_env.yaml`, so it reloads on another machine with its
> environment); the `.joblib` serves **deployment**, and is what goes into the API's
> Docker image.

> The `/predict` API (Part 3) reloads that `.joblib` as-is and hands it **raw**
> features — the whole preprocessing travels with the model. It is fitted with the
> model, serialised with the model, deployed with the model. Separating them would
> create two artifacts that must stay in sync manually, and would not.

## Tests

```bash
cd model
pip install pytest
pytest -q
```

Eight tests on the pure functions — the ones that only compute. Cleaning removes
exactly the three implausible rows; the declared feature groups cover every column
of the dataset (one left out would silently stop being used, with no error
anywhere); the preprocessor still produces the right shape for a brand it never
saw, which pins the `handle_unknown="ignore"` the API depends on; and the metrics
behave, including RMSE never falling below MAE.

`main()` is deliberately left out. Testing it would mean replacing MLflow and the
filesystem with stand-ins, and what remains to assert once you have is that the
stand-ins were called — a test of the mock rather than of the code.

## Extensions

- **Model registry** — immediately available now that the backend is a SQL database
  (it does not work with the local file store). All that is missing is a
  `mlflow.register_model()` and a Staging → Production promotion policy; the API
  would then ask the registry for "the Production version" instead of embedding a
  frozen file in its image. The embedded model was preferred here: no frequent
  retraining, a single consumer, and no runtime dependency on the tracking stack.
- **Permutation importance** — impurity-based `feature_importances_` is biased
  towards high-cardinality features, which after one-hot encoding is exactly what
  `model_key` is.
- **Production monitoring** — a different problem altogether: no longer comparing
  training runs but watching a model serve real requests (data drift, performance
  decay, latency). It would require **logging the API's predictions**, which it does
  not do today, and finding a business proxy for quality — the "true" optimal price
  is never observed after the fact.

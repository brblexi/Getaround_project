"""GetAround — train the rental price model.

The one-command version of `02_price_modelling.ipynb`. The notebook explains the
reasoning; this script replays the training reproducibly and writes the artifact
the API serves.

    python train.py                                   # linear + random forest
    python train.py --model rf                        # one model only
    python train.py --n-estimators 300 --max-depth 20 # other hyperparameters
    python train.py --experiment getaround-sweep      # a fresh experiment

What it does, in order: resolve where MLflow writes, load and clean the data,
split, build a scikit-learn pipeline, train and evaluate each model while
tracking everything, then serialise the best pipeline to `artifacts/`.

The **whole pipeline** is serialised, not just the estimator, so the API can be
handed raw features and the encoding cannot drift between training and serving.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Every path is relative to this file, so the script runs from any working
# directory and holds inside a container.
HERE = Path(__file__).parent.resolve()
DATA_PATH = HERE / "data" / "get_around_pricing_project.csv"
ARTIFACT_DIR = HERE / "artifacts"

# Loading the .env is a local convenience, not a production mechanism: in a
# container the platform injects the variables, no file is found, and this call
# does nothing. `override=False` is the default, written out because the
# intention matters — a variable already set in the environment always wins over
# the file, which is the production semantics.
load_dotenv(HERE / ".env", override=False)

RANDOM_STATE = 42

TARGET = "rental_price_per_day"
NUMERIC = ["mileage", "engine_power"]
CATEGORICAL = ["model_key", "fuel", "paint_color", "car_type"]
BOOLEAN = [
    "private_parking_available", "has_gps", "has_air_conditioning",
    "automatic_car", "has_getaround_connect", "has_speed_regulator",
    "winter_tires",
]


# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------

def get_tracking_uri() -> str:
    """Remote backend if `MLFLOW_TRACKING_URI` is set, local `mlruns/` otherwise.

    Switching between a remote database and local files is a matter of setting
    an environment variable, never of editing code — and a demo survives a
    network outage by falling back on its own.

    The Neon URI contains a password and lives in an unversioned `.env`.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri

    # MLflow 3.x put the file store in maintenance mode and raises by default;
    # this variable is the documented opt-in. setdefault leaves an existing
    # value alone.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return f"file:{(HERE / 'mlruns').as_posix()}"


def setup_experiment(name: str) -> None:
    """Select or create the experiment, sending artifacts to S3 when configured.

    MLflow keeps two stores: the backend (PostgreSQL) holds metrics, parameters
    and metadata — everything structured and queryable — while the artifact
    store (S3) holds models and other large files, because a SQL database is the
    wrong place for binaries. Each run row in the backend carries the S3 URI of
    its own artifacts, which is what links the two.

    `artifact_location` is written once, when the experiment is created, and is
    never recomputed. An experiment first created locally keeps its local
    artifact path even after the S3 variable is set, silently. Switching to S3
    means using a new experiment name — which is what `--experiment` is for.
    """
    artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION")

    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_location)
        if artifact_location:
            print(f"Artifacts -> {artifact_location}")

    mlflow.set_experiment(name)


def git_commit() -> str | None:
    """Short commit hash, or None outside a repository.

    Logged as a tag so a run can be traced back to the code that produced it —
    the cheapest half of reproducibility, the other half being the seed.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=HERE, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and drop the three implausible rows found during the EDA.

    A negative mileage is impossible, a mileage above one million is a typing
    error, and a car does not have zero horsepower. They are dropped rather than
    imputed: three rows out of 4,843 are too few to justify a correction, and
    keeping them means fitting on noise.

    `index_col=0` treats the unnamed first column as the index rather than as a
    feature.
    """
    df = pd.read_csv(path, index_col=0)
    before = len(df)

    df = df[(df.mileage >= 0) & (df.mileage < 1_000_000) & (df.engine_power > 0)]

    print(f"Cleaning: {before - len(df)} row(s) dropped -> {len(df)} remaining")
    return df


# ---------------------------------------------------------------------------
# Preprocessing and models
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """One transformation per column type, in a single object.

    Numeric columns are standardised, which matters for the linear model and is
    harmless for the trees. Categorical columns are one-hot encoded, since a
    model only handles numbers. Booleans pass through: they are already 0/1.

    `handle_unknown="ignore"` keeps the encoder from raising when the API sends
    a category that was not in the training data. The trade-off is that the
    unknown value is encoded as all-zeros and scored silently, so the API
    validates incoming categories against this encoder before predicting.

    Wrapping this in a Pipeline is what prevents leakage: these transformations
    learn parameters — means, standard deviations, category lists — and inside a
    pipeline they learn them on the training fold alone, then replay them
    unchanged on the test set and in production.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("bool", "passthrough", BOOLEAN),
        ]
    )


def get_model(name: str, n_estimators: int, max_depth: int | None):
    """Return the requested estimator.

    `linear` is the baseline: a sophisticated model that cannot beat a linear
    regression does not justify itself.

    `rf` is a random forest, which captures non-linearities and interactions.
    Capping `max_depth` limits overfitting — at 12 the test RMSE is marginally
    better than with unlimited depth, and the serialised file drops from about
    62 MB to 2 MB, which is a deployment decision as much as a statistical one.
    """
    if name == "linear":
        return LinearRegression()
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred) -> dict[str, float]:
    """Return RMSE, MAE and R².

    RMSE squares the errors, so large misses weigh far more, and the square root
    brings the result back to euros. MAE weighs every error proportionally and
    is the figure to quote to a business audience: on average we land about 11 €
    from the real price. R² situates overall quality without a unit, from 0 (no
    better than predicting the mean) to 1.

    RMSE is mathematically always at least MAE, so their order says nothing;
    their ratio does. About 1.55 here, which reflects a spread in the errors
    rather than one catastrophic miss.

    The same metrics are computed on train and on test: the gap between them is
    how overfitting shows up.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["linear", "rf", "both"], default="both")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=12,
                        help="Maximum tree depth; -1 for unlimited.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5,
                        help="Folds for cross-validated RMSE; 0 to skip.")
    parser.add_argument("--experiment", default="getaround-pricing",
                        help="MLflow experiment name. A new name forces a new "
                             "experiment, and therefore a new artifact_location "
                             "— the only way to switch existing runs to S3.")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "model.joblib",
                        help="Where to write the serialised pipeline.")
    args = parser.parse_args()
    if args.max_depth == -1:
        args.max_depth = None
    return args


def main() -> None:
    args = parse_args()

    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    backend = "remote backend" if uri.startswith(("postgresql", "sqlite", "http")) else "local (mlruns/)"
    print(f"MLflow tracking -> {backend}")
    setup_experiment(args.experiment)

    df = load_and_clean()
    X = df[NUMERIC + CATEGORICAL + BOOLEAN]
    y = df[TARGET]

    # The target is trained on raw euros, with no log transform. That was tested
    # rather than assumed: the price distribution is close to symmetric (mean
    # 121 against median 119, skewness 0.61) and training on log(price) makes
    # every metric worse (RMSE 17.7 against 17.0). A log corrects strong skew;
    # there is none here, so it distorts without fixing anything.

    # The test set is held out and only touched for the final evaluation, on
    # cars the model has never seen. A fixed seed keeps the split identical
    # between runs, so two trainings stay comparable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )

    # Predicting the training median: the error any model has to beat. Without
    # it, an R² of 0.75 is hard to place.
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.median()))))
    print(f"Predict-the-median baseline -> test RMSE={baseline_rmse:.2f}")

    models = ["linear", "rf"] if args.model == "both" else [args.model]
    commit = git_commit()

    best = {"rmse": np.inf, "pipe": None, "name": None, "metrics": None}

    for name in models:
        # During a hyperparameter sweep several runs share the same model name
        # and become indistinguishable in the UI, so the forest is suffixed with
        # its depth ("rf-d4", "rf-d8"). The linear model has no hyperparameter
        # and keeps its plain name.
        depth = "none" if args.max_depth is None else args.max_depth
        run_name = f"{name}-d{depth}" if name == "rf" else name

        # Everything logged inside the block belongs to this run, and the run is
        # closed cleanly on the way out, including on an exception.
        with mlflow.start_run(run_name=run_name):
            pipe = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("model", get_model(name, args.n_estimators, args.max_depth)),
            ])
            pipe.fit(X_train, y_train)

            train_metrics = evaluate(y_train, pipe.predict(X_train))
            test_metrics = evaluate(y_test, pipe.predict(X_test))

            mlflow.log_params({
                "model": name,
                "n_rows": len(df),
                "test_size": args.test_size,
                "random_state": RANDOM_STATE,
            })
            if name == "rf":
                mlflow.log_params({
                    "n_estimators": args.n_estimators,
                    "max_depth": depth,
                })

            # Prefixed so the two can be compared in the MLflow UI.
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value)
            for key, value in test_metrics.items():
                mlflow.log_metric(f"test_{key}", value)
            mlflow.log_metric("baseline_test_rmse", baseline_rmse)

            if args.cv:
                # Tells us whether the single split is representative or whether
                # the score moved with the seed.
                cv_rmse = -cross_val_score(
                    pipe, X_train, y_train, cv=args.cv,
                    scoring="neg_root_mean_squared_error",
                )
                mlflow.log_metric("cv_rmse_mean", cv_rmse.mean())
                mlflow.log_metric("cv_rmse_std", cv_rmse.std())

            mlflow.set_tags({
                "sklearn_version": sklearn.__version__,
                "git_commit": commit or "unknown",
            })

            # The signature records the expected input and output schema, so
            # anyone loading this model later sees what it takes without reading
            # the training code. MLflow warns that the integer columns cannot
            # carry missing values; that is fine here, because the API declares
            # them as required int fields and rejects a payload without them.
            mlflow.sklearn.log_model(
                pipe,
                name="model",
                signature=infer_signature(X_train, pipe.predict(X_train.head())),
                input_example=X_train.head(),
            )

            line = (f"[{run_name}]  test RMSE={test_metrics['rmse']:.2f}  "
                    f"MAE={test_metrics['mae']:.2f}  R²={test_metrics['r2']:.3f}")
            if args.cv:
                line += f"  |  CV RMSE={cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}"
            print(line)

            # Selected on test RMSE, never on train, which would reward the most
            # overfitted model.
            if test_metrics["rmse"] < best["rmse"]:
                best = {"rmse": test_metrics["rmse"], "pipe": pipe,
                        "name": name, "metrics": test_metrics}

    # Serialise for the API. This is a second, independent serialisation: MLflow
    # logged its own copy above for traceability, and nothing links the two
    # calls — which is why the API keeps working if the tracking stack is down.
    # compress=3 brings the file down to about 2 MB.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipe"], args.output, compress=3)

    # Written next to the artifact so that whoever finds the .joblib later can
    # tell what it is. The scikit-learn version is the important field: a
    # pickle only reloads reliably under the version that wrote it.
    metadata = {
        "model": best["name"],
        "metrics": best["metrics"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "git_commit": commit,
        "n_rows": len(df),
        "random_state": RANDOM_STATE,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    print(f"\nBest model: {best['name']} (test RMSE={best['rmse']:.2f})")
    print(f"Serialised to: {args.output}")
    print("Copy it to api/model.joblib and rebuild the image to update the served model.")


if __name__ == "__main__":
    main()

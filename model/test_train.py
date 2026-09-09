"""Tests for the training script.

Run from `model/`:  pytest -q

Only the pure functions are tested — cleaning, preprocessing, model selection,
metrics. `main()` is not: it writes to MLflow and to disk, and what would be
left to assert after mocking all of that is that the calls happen in order,
which is a test of the mock rather than of the code.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import train


@pytest.fixture
def raw_csv(tmp_path):
    """A small valid dataset with the three implausible rows appended."""
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({
        "model_key": rng.choice(["Citroën", "Renault"], n),
        "mileage": rng.integers(1_000, 200_000, n),
        "engine_power": rng.integers(60, 250, n),
        "fuel": rng.choice(["diesel", "petrol"], n),
        "paint_color": rng.choice(["black", "white"], n),
        "car_type": rng.choice(["sedan", "suv"], n),
        **{col: rng.random(n) > 0.5 for col in train.BOOLEAN},
        "rental_price_per_day": rng.integers(80, 200, n),
    })
    df.loc[0, "mileage"] = -50
    df.loc[1, "mileage"] = 1_500_000
    df.loc[2, "engine_power"] = 0

    path = tmp_path / "pricing.csv"
    df.to_csv(path)
    return path


def test_cleaning_drops_exactly_the_implausible_rows(raw_csv):
    df = train.load_and_clean(raw_csv)

    assert len(df) == 47
    assert (df.mileage >= 0).all()
    assert (df.mileage < 1_000_000).all()
    assert (df.engine_power > 0).all()


def test_the_feature_groups_cover_the_dataset(raw_csv):
    """A column missing from the three lists is silently dropped from training.

    Nothing would fail — the model would just quietly stop using it — so this
    check is the only place that would notice.
    """
    df = train.load_and_clean(raw_csv)
    declared = set(train.NUMERIC + train.CATEGORICAL + train.BOOLEAN + [train.TARGET])

    assert declared == set(df.columns)


def test_the_preprocessor_handles_an_unseen_category(raw_csv):
    """`handle_unknown="ignore"` is what keeps the API from raising on a brand
    it has never seen. If this default were ever changed, the service would
    start returning 500s in production instead of a price."""
    df = train.load_and_clean(raw_csv)
    preprocessor = train.build_preprocessor()
    preprocessor.fit(df)

    unseen = df.head(1).copy()
    unseen["model_key"] = "Delorean"

    encoded = preprocessor.transform(unseen)
    assert encoded.shape[1] == preprocessor.transform(df.head(1)).shape[1]


def test_get_model_returns_the_requested_estimator():
    assert isinstance(train.get_model("linear", 10, 5), LinearRegression)

    forest = train.get_model("rf", n_estimators=10, max_depth=5)
    assert isinstance(forest, RandomForestRegressor)
    assert forest.n_estimators == 10
    assert forest.max_depth == 5
    assert forest.random_state == train.RANDOM_STATE  # reproducibility


def test_get_model_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        train.get_model("xgboost", 10, 5)


def test_a_perfect_prediction_scores_perfectly():
    y = pd.Series([100.0, 120.0, 140.0])
    metrics = train.evaluate(y, y)

    assert metrics == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}


def test_rmse_is_never_below_mae():
    """A mathematical property, and the reason their ratio is what informs
    rather than their order — worth pinning so the claim in the README stays
    true if the metric code is ever touched."""
    rng = np.random.default_rng(1)
    y_true = pd.Series(rng.normal(120, 30, 200))
    y_pred = y_true + rng.normal(0, 15, 200)

    metrics = train.evaluate(y_true, y_pred)
    assert metrics["rmse"] >= metrics["mae"]


def test_the_pipeline_trains_and_predicts_end_to_end(raw_csv):
    df = train.load_and_clean(raw_csv)
    pipe = Pipeline([
        ("preprocessor", train.build_preprocessor()),
        ("model", train.get_model("rf", n_estimators=5, max_depth=3)),
    ])
    pipe.fit(df, df[train.TARGET])

    # Raw features in, one price out — the contract the API depends on.
    prediction = pipe.predict(df.head(1))
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])

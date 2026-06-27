"""
=============================================================================
 GetAround — Entraînement du modèle de prédiction du prix de location
=============================================================================
Pipeline complet, exécutable en une commande :

    python train.py                 # entraîne Linéaire + RandomForest
    python train.py --model rf      # un seul modèle
    python train.py --n-estimators 300 --max-depth 20

Ce que fait le script :
  1. charge et NETTOIE les données (3 lignes aberrantes retirées) ;
  2. construit un PIPELINE scikit-learn (prétraitement + modèle) ;
  3. entraîne, évalue (RMSE / MAE / R²) ;
  4. trace tout dans MLflow (paramètres, métriques, modèle) ;
  5. sérialise le MEILLEUR modèle dans artifacts/model.joblib
     -> c'est CE fichier que l'API /predict réutilisera en Partie 3.
=============================================================================
"""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Chemins, construits en relatif par rapport à ce fichier (portable) ---
HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "get_around_pricing_project.csv"
ARTIFACT_DIR = HERE / "artifacts"
TARGET = "rental_price_per_day"

# Colonnes par type (sert à aiguiller le prétraitement).
NUMERIC = ["mileage", "engine_power"]
CATEGORICAL = ["model_key", "fuel", "paint_color", "car_type"]
BOOLEAN = ["private_parking_available", "has_gps", "has_air_conditioning",
           "automatic_car", "has_getaround_connect", "has_speed_regulator",
           "winter_tires"]


def get_tracking_uri() -> str:
    """
    Choisit OÙ MLflow enregistre les runs, sans modifier le reste du code :
      - si la variable d'environnement MLFLOW_TRACKING_URI est définie
        (ex. une base PostgreSQL/Neon) -> on l'utilise ;
      - sinon -> repli sur le dossier local mlruns/ (filet de sécurité démo).

    On bascule donc base distante <-> local en (dé)définissant une simple
    variable d'environnement, jamais en touchant au code. L'URL Neon contient
    un mot de passe : elle vit dans un fichier .env NON versionné, pas ici.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    # Repli local : MLflow 3.x exige cet opt-in pour le file store déprécié.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return f"file:{(HERE / 'mlruns').as_posix()}"


def setup_experiment(name: str = "getaround-pricing") -> None:
    """
    Sélectionne (ou crée) l'expérience MLflow, et choisit OÙ vont les ARTEFACTS
    (le modèle sérialisé, les fichiers loggés) :
      - si MLFLOW_ARTIFACT_LOCATION est défini (ex. 's3://bucket/mlflow-artifacts')
        -> les artefacts sont stockés sur S3 (via boto3 + identifiants AWS) ;
      - sinon -> emplacement par défaut (local).

    ⚠️ L'artifact_location est FIGÉ à la CRÉATION de l'expérience. Pour basculer
    vers S3, l'expérience doit être neuve : si elle existe déjà (créée en local),
    elle conserve son ancien emplacement d'artefacts.

    Rappel du partage des rôles :
      - backend store (PostgreSQL/Neon) -> métriques, paramètres, métadonnées ;
      - artifact store (S3)             -> modèles et fichiers lourds.
    """
    artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION")  # None -> défaut local
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        mlflow.create_experiment(name, artifact_location=artifact_location)
        if artifact_location:
            print(f"Artefacts -> {artifact_location}")
    mlflow.set_experiment(name)


def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge le CSV et retire les 3 lignes aberrantes repérées à l'EDA."""
    df = pd.read_csv(path, index_col=0)
    before = len(df)
    # mileage négatif (impossible), mileage absurde (>1M km), moteur à 0 ch.
    df = df[(df.mileage >= 0) & (df.mileage < 1_000_000) & (df.engine_power > 0)]
    print(f"Nettoyage : {before - len(df)} ligne(s) retirée(s) -> {len(df)} restantes")
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Prétraitement par type de colonne, en un seul objet :
      - numériques   -> StandardScaler (centre/réduit)
      - catégorielles-> OneHotEncoder (handle_unknown='ignore' : indispensable
                        pour que l'API ne plante pas sur une modalité inconnue)
      - booléennes   -> telles quelles (déjà 0/1)
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("bool", "passthrough", BOOLEAN),
        ]
    )


def get_model(name: str, n_estimators: int, max_depth):
    """Renvoie l'estimateur demandé."""
    if name == "linear":
        return LinearRegression()
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, n_jobs=-1,
        )
    raise ValueError(f"Modèle inconnu : {name}")


def evaluate(y_true, y_pred) -> dict:
    """RMSE, MAE, R² regroupés dans un dict."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["linear", "rf", "both"], default="both")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    # MLflow : où écrire les runs (base distante si MLFLOW_TRACKING_URI, sinon
    # dossier local mlruns/) + sélection de l'expérience (artefacts -> S3 si défini).
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    backend = "base distante" if uri.startswith(("postgresql", "sqlite", "http")) else "local (mlruns/)"
    print(f"MLflow tracking -> {backend}")
    setup_experiment("getaround-pricing")

    df = load_and_clean()
    X = df[NUMERIC + CATEGORICAL + BOOLEAN]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    models = ["linear", "rf"] if args.model == "both" else [args.model]
    best = {"rmse": np.inf, "pipe": None, "name": None}

    for name in models:
        # Un run MLflow par modèle : il enregistre params + métriques + modèle.
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("model", get_model(name, args.n_estimators, args.max_depth)),
            ])
            pipe.fit(X_train, y_train)

            train_m = evaluate(y_train, pipe.predict(X_train))
            test_m = evaluate(y_test, pipe.predict(X_test))

            # --- Traçage MLflow ---
            mlflow.log_param("model", name)
            mlflow.log_param("n_rows", len(df))
            if name == "rf":
                mlflow.log_param("n_estimators", args.n_estimators)
                mlflow.log_param("max_depth", args.max_depth)
            for k, v in train_m.items():
                mlflow.log_metric(f"train_{k}", v)
            for k, v in test_m.items():
                mlflow.log_metric(f"test_{k}", v)
            mlflow.sklearn.log_model(pipe, name="model")

            print(f"\n[{name}]  test RMSE={test_m['rmse']:.2f}  "
                  f"MAE={test_m['mae']:.2f}  R²={test_m['r2']:.3f}")

            if test_m["rmse"] < best["rmse"]:
                best = {"rmse": test_m["rmse"], "pipe": pipe, "name": name}

    # --- Sérialisation du meilleur modèle pour l'API (Partie 3) ---
    ARTIFACT_DIR.mkdir(exist_ok=True)
    out = ARTIFACT_DIR / "model.joblib"
    joblib.dump(best["pipe"], out, compress=3)  # compress=3 : modèle léger (~2 Mo)
    print(f"\n✅ Meilleur modèle : {best['name']} (RMSE={best['rmse']:.2f})")
    print(f"   Sérialisé dans : {out}")


if __name__ == "__main__":
    main()

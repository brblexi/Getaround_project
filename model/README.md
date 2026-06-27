# Partie 2 — Modèle de prédiction du prix de location

Prédit `rental_price_per_day` à partir des caractéristiques d'une voiture.
Régression supervisée, suivi des expériences avec **MLflow**.

## Contenu

```
model/
├── 02_modelisation_pricing.ipynb   # notebook narratif (démarche complète)
├── train.py                        # version exécutable en une commande
├── data/                           # jeu de données pricing
├── artifacts/model.joblib          # pipeline entraîné (réutilisé par l'API)
└── requirements.txt
```

## Démarche

Nettoyage (3 lignes aberrantes retirées) → pipeline scikit-learn
(`StandardScaler` + `OneHotEncoder` + booléens) → entraînement de
`LinearRegression` (baseline) vs `RandomForestRegressor` → suivi MLflow →
sérialisation du meilleur modèle.

**Résultats (jeu de test) :** le RandomForest l'emporte avec RMSE ≈ 17 €,
MAE ≈ 11 €, R² ≈ 0,75 (baseline linéaire : RMSE ≈ 18 €, R² ≈ 0,70).

## Lancer l'entraînement

```bash
pip install -r requirements.txt
python train.py                 # entraîne les 2 modèles, log MLflow, sauvegarde le meilleur
python train.py --model rf      # un seul modèle
python train.py --n-estimators 300 --max-depth 20   # autres hyperparamètres
```

## Visualiser les expériences MLflow

Le suivi **bascule automatiquement** selon les variables d'environnement
(`get_tracking_uri()` + `setup_experiment()`), avec une séparation
**backend store / artifact store** :

| Donnée | Va dans | Variable |
|---|---|---|
| Métriques, paramètres, métadonnées des runs | **PostgreSQL / Neon** | `MLFLOW_TRACKING_URI` |
| Modèle sérialisé, fichiers loggés | **S3** | `MLFLOW_ARTIFACT_LOCATION` (+ clés AWS) |

Si une variable est absente → **repli local** (`mlruns/`).

**En local** (aucune variable) :

```bash
mlflow ui      # http://localhost:5000
```

**Avec Neon + S3** : copier `.env.example` en `.env`, le remplir, puis :

```bash
# charge les variables du .env dans le shell (ou via python-dotenv)
export $(grep -v '^#' .env | xargs)
python train.py                                        # métriques -> Neon, modèle -> S3
mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"   # UI branchée sur Neon
```

> 🔐 **Secrets** : l'URL Neon (mot de passe) et les clés AWS vivent dans `.env`
> (non versionné, voir `.gitignore`) en local, et dans les *secrets* du Space
> Hugging Face en ligne — jamais en dur dans le code ni sur GitHub.

> 📦 **Pourquoi deux stockages ?** Une base SQL (Neon) est faite pour la donnée
> structurée requêtable (chiffres, texte), pas pour des fichiers binaires ; S3
> est fait pour les fichiers. Neon conserve, pour chaque run, l'URI S3 de ses
> artefacts pour faire le lien.

> Le pipeline **complet** (prétraitement + modèle) est aussi sérialisé dans
> `artifacts/model.joblib`. L'API `/predict` (Partie 3) le recharge tel quel et
> lui passe des caractéristiques **brutes** — tout le prétraitement est embarqué.

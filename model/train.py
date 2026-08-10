"""
=============================================================================
 GetAround — Entraînement du modèle de prédiction du prix de location
=============================================================================

RÔLE DE CE FICHIER
------------------
C'est la version EXÉCUTABLE EN UNE COMMANDE de la démarche racontée pas à pas
dans le notebook `02_modelisation_pricing.ipynb`. Le notebook sert à
comprendre et à présenter ; ce script sert à REJOUER l'entraînement de façon
reproductible — c'est l'esprit "industrialisation" du bloc 5.

    python train.py                              # entraîne Linéaire + RandomForest
    python train.py --model rf                   # un seul modèle
    python train.py --n-estimators 300 --max-depth 20   # autres hyperparamètres

CE QUE FAIT LE SCRIPT, DANS L'ORDRE
-----------------------------------
  1. choisit OÙ MLflow enregistre (base distante Neon / S3, ou repli local) ;
  2. charge et NETTOIE les données (3 lignes aberrantes retirées) ;
  3. découpe train / test ;
  4. construit un PIPELINE scikit-learn (prétraitement + modèle) ;
  5. entraîne chaque modèle, l'évalue (RMSE / MAE / R²) et TRACE tout dans MLflow ;
  6. sérialise le MEILLEUR modèle dans artifacts/model.joblib
     -> c'est CE fichier que l'API /predict recharge en Partie 3.

POURQUOI UN PIPELINE ET PAS JUSTE UN MODÈLE
--------------------------------------------------------------------
On sérialise le pipeline COMPLET (prétraitement + modèle). Ainsi l'API reçoit
des caractéristiques BRUTES ("Citroën", "diesel", 140000 km) et l'encodage
s'applique tout seul, exactement comme à l'entraînement. Si on ne sauvegardait
que le RandomForest, il faudrait redupliquer toute la logique d'encodage côté
API -> source d'incohérences ("train/serving skew").
=============================================================================
"""
# --- Bibliothèque standard ---------------------------------------------------
import argparse          # lecture des options passées en ligne de commande
import os                # accès aux variables d'environnement (secrets, config)
from pathlib import Path # manipulation de chemins, portable Windows/Linux/macOS

# --- Bibliothèques tierces ---------------------------------------------------
import joblib            # sérialisation (écrire/relire le modèle sur disque)
import numpy as np
import pandas as pd
import mlflow            # suivi des expériences
import mlflow.sklearn    # sous-module : sait logger un modèle scikit-learn
from dotenv import load_dotenv   # lecture OPTIONNELLE d'un .env local

from sklearn.compose import ColumnTransformer      # traitement PAR TYPE de colonne
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline              # enchaîne prétraitement -> modèle
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# CONSTANTES
# =============================================================================

# Path(__file__) = le chemin de CE fichier ; .parent = le dossier qui le contient.
# On construit donc tous les chemins EN RELATIF par rapport au script lui-même :
# il fonctionne quel que soit le dossier depuis lequel on le lance, et sans
# aucun chemin absolu en dur (portable : machine locale, conteneur Docker...).
HERE = Path(__file__).parent

# --- Chargement du .env : COMMODITÉ LOCALE, PAS UN MÉCANISME DE PRODUCTION ----
# En production (conteneur Docker, Space Hugging Face, ECS...), les variables
# d'environnement sont INJECTÉES PAR LA PLATEFORME : il n'y a pas de fichier
# .env, et cet appel ne trouve rien -> il ne fait simplement rien, sans erreur.
# En local, il évite d'avoir à retaper les variables à chaque nouveau terminal.
#
# override=False (le défaut, écrit ici pour que l'intention soit explicite) :
# une variable DÉJÀ présente dans l'environnement N'EST PAS écrasée par le .env.
# C'est la sémantique de production : la plateforme a toujours le dernier mot,
# et un `$env:MLFLOW_TRACKING_URI = ...` posé à la main pour un test ponctuel
# prime sur le fichier.
#
# Le code métier, lui, ne lit QUE os.environ : il ignore d'où viennent les
# valeurs. C'est ce qui rend le script identique en local et en production.
load_dotenv(HERE / ".env", override=False)

DATA_PATH = HERE / "data" / "get_around_pricing_project.csv"
ARTIFACT_DIR = HERE / "artifacts"

# La CIBLE : la variable que le modèle doit apprendre à prédire.
TARGET = "rental_price_per_day"

# Les FEATURES, regroupées par type. Ce découpage n'est pas cosmétique :
# c'est lui qui aiguille chaque colonne vers le bon prétraitement (voir plus bas).
NUMERIC = ["mileage", "engine_power"]
CATEGORICAL = ["model_key", "fuel", "paint_color", "car_type"]
BOOLEAN = ["private_parking_available", "has_gps", "has_air_conditioning",
           "automatic_car", "has_getaround_connect", "has_speed_regulator",
           "winter_tires"]


# =============================================================================
# 1. CONFIGURATION MLFLOW (où atterrissent les runs et les artefacts)
# =============================================================================

def get_tracking_uri() -> str:
    """
    Choisit OÙ MLflow enregistre les runs, SANS modifier le reste du code :
      - si la variable d'environnement MLFLOW_TRACKING_URI est définie
        (ex. une base PostgreSQL/Neon) -> on l'utilise ;
      - sinon -> repli sur le dossier local mlruns/ (filet de sécurité démo).

    L'intérêt : on bascule base distante <-> local en (dé)définissant une simple
    variable d'environnement, jamais en touchant au code. Et le jour où le
    réseau lâche pendant une démo, on retombe automatiquement en local.

    SECRET : l'URL Neon contient un mot de passe. Elle vit dans un fichier
    .env NON versionné (voir .gitignore et .env.example), jamais ici.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")   # None si la variable n'existe pas
    if uri:
        return uri

    # Repli local. MLflow 3.x a mis le "file store" (dossier mlruns/) en mode
    # maintenance et lève une erreur par défaut ; cette variable est l'opt-in
    # officiel pour continuer à l'utiliser. setdefault = "ne pose la valeur que
    # si elle n'est pas déjà définie" (on n'écrase pas un réglage existant).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    # Le préfixe "file:" indique à MLflow qu'il s'agit d'un dossier local.
    # as_posix() normalise le chemin avec des "/" (utile sous Windows).
    return f"file:{(HERE / 'mlruns').as_posix()}"


def setup_experiment(name: str = "getaround-pricing") -> None:
    """
    Sélectionne (ou crée) l'expérience MLflow et choisit OÙ vont les ARTEFACTS
    (le modèle loggé, les fichiers) :
      - si MLFLOW_ARTIFACT_LOCATION est défini (ex. 's3://bucket/mlflow-artifacts')
        -> les artefacts partent sur S3 (via boto3 + identifiants AWS) ;
      - sinon -> emplacement par défaut (local).

    RAPPEL DU PARTAGE DES RÔLES (MLflow sépare DEUX stockages) :
      - backend store  (PostgreSQL/Neon) -> métriques, paramètres, métadonnées
        = tout ce qui est structuré et requêtable, "ce qui tient dans un tableur" ;
      - artifact store (S3)              -> modèles et fichiers lourds,
        parce qu'une base SQL n'est pas faite pour stocker du binaire.
      Le backend conserve, pour chaque run, l'URI S3 de ses artefacts : c'est
      ce qui fait le lien entre les deux.

    ⚠️ PIÈGE À CONNAÎTRE : artifact_location est FIGÉ À LA CRÉATION de
    l'expérience. Si l'expérience existe déjà (créée en local), elle GARDE son
    ancien emplacement d'artefacts même si tu définis la variable S3 ensuite.
    Pour basculer vraiment vers S3, il faut une expérience neuve.
    """
    
    # On lit la variable d'environnement. .get() renvoie None si elle n'existe pas
    # (contrairement à os.environ["..."] qui lèverait une erreur).
    # si None -> MLflow utilisera son emplacement d'artefacts par défaut (local).
    artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION")

    # On demande à MLflow : "cette EXP existe-t-elle déjà dans Neon (une ligne avec ce name) ?"
    # Réponse None → create_experiment fait l'INSERT de cette EXP dans Neon, avec l'artifact_location S3. 
    # Puis set_experiment sélectionne cette EXP pour les runs qui suivent
    exp = mlflow.get_experiment_by_name(name)

    if exp is None:                    # elle n'existe pas -> on la crée
        mlflow.create_experiment(name, artifact_location=artifact_location)
        if artifact_location:          # on ne l'affiche que si S3 est configuré
            print(f"Artefacts -> {artifact_location}")

    # Dans les deux cas (créée à l'instant OU déjà existante), on l'ouvre.
    mlflow.set_experiment(name)


# =============================================================================
# 2. DONNÉES : chargement et nettoyage
# =============================================================================

def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Charge le CSV et retire les 3 lignes aberrantes repérées pendant l'EDA.

    index_col=0 : la première colonne du fichier est un index sans nom
    (une simple numérotation des voitures), on l'utilise comme index plutôt
    que de la traiter comme une variable explicative.
    """
    df = pd.read_csv(path, index_col=0)
    before = len(df)

    # Les 3 anomalies identifiées à l'exploration (0,06 % des données) :
    #   - un mileage NÉGATIF        -> physiquement impossible ;
    #   - un mileage > 1 000 000 km -> valeur absurde (erreur de saisie) ;
    #   - un engine_power = 0       -> une voiture n'a pas 0 cheval.
    # On les supprime plutôt que de les imputer : elles sont trop peu nombreuses
    # pour justifier une correction, et les garder ferait apprendre du bruit.
    df = df[(df.mileage >= 0) & (df.mileage < 1_000_000) & (df.engine_power > 0)]

    print(f"Nettoyage : {before - len(df)} ligne(s) retirée(s) -> {len(df)} restantes")
    return df


# =============================================================================
# 3. PRÉTRAITEMENT ET MODÈLES
# =============================================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Prétraitement par TYPE de colonne, regroupé dans un seul objet.

      - numériques    -> StandardScaler : centre (moyenne 0) et réduit (écart-type 1).
                         Utile surtout pour la régression linéaire, qui est
                         sensible aux échelles ; sans effet néfaste sur les arbres.

      - catégorielles -> OneHotEncoder : transforme "diesel"/"petrol"... en
                         colonnes binaires (une par modalité), car un modèle ne
                         sait manipuler que des nombres.
                         handle_unknown="ignore" est CRUCIAL en production :
                         si l'API reçoit une marque jamais vue à l'entraînement,
                         l'encodeur met des 0 partout au lieu de LEVER UNE ERREUR
                         et de faire planter le service.

      - booléennes    -> "passthrough" : laissées telles quelles, elles sont
                         déjà en 0/1, il n'y a rien à transformer.

    ⚠️ POURQUOI DANS UN PIPELINE : ces transformations APPRENNENT des paramètres
    sur les données (moyenne/écart-type pour le scaler, liste des modalités pour
    l'encodeur). Dans un Pipeline, cet apprentissage se fait sur le TRAIN seul,
    puis est réappliqué à l'identique au test et en production -> pas de fuite
    de données (data leakage).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("bool", "passthrough", BOOLEAN),
        ]
    )


def get_model(name: str, n_estimators: int, max_depth):
    """
    Renvoie l'estimateur demandé.

      - "linear" : LinearRegression, la BASELINE. Elle sert de référence :
        c'est le minimum à battre. Un modèle sophistiqué qui ne bat pas une
        régression linéaire ne se justifie pas.

      - "rf" : RandomForestRegressor, une forêt d'arbres de décision. Capte les
        relations NON LINÉAIRES et les interactions entre variables.
          * n_estimators = nombre d'arbres (plus = plus stable, mais plus lourd) ;
          * max_depth    = profondeur maximale de chaque arbre. La plafonner
            LIMITE LE SURAPPRENTISSAGE : sans limite, les arbres mémorisent le
            train. Ici max_depth=12 donne un RMSE de test légèrement MEILLEUR
            qu'en illimité, et fait passer le fichier de 62 Mo à ~2 Mo ;
          * random_state=42 : fige l'aléa (tirage des échantillons et des
            variables) -> résultats REPRODUCTIBLES d'une exécution à l'autre ;
          * n_jobs=-1 : utilise tous les cœurs du processeur (entraînement plus rapide).
    """
    if name == "linear":
        return LinearRegression()
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, n_jobs=-1,
        )
    raise ValueError(f"Modèle inconnu : {name}")


# =============================================================================
# 4. ÉVALUATION
# =============================================================================

def evaluate(y_true, y_pred) -> dict:
    """
    Calcule les trois métriques de régression, regroupées dans un dictionnaire.

      - RMSE : racine de la moyenne des erreurs AU CARRÉ. Le carré fait que les
        GROSSES erreurs pèsent beaucoup plus ; la racine ramène le résultat en
        euros, donc lisible. C'est la métrique de référence en régression.

      - MAE : moyenne des erreurs en VALEUR ABSOLUE. Toutes les erreurs comptent
        proportionnellement. C'est la plus parlante pour le métier :
        "en moyenne, on tombe à ~11 € du vrai prix".

      - R² : proportion de variance expliquée, de 0 (= ne fait pas mieux que
        prédire la moyenne) à 1 (= parfait). Sans unité, il situe la qualité globale.

    À SAVOIR : la RMSE est TOUJOURS >= la MAE (propriété mathématique). Ce n'est
    donc pas leur ordre qui informe, mais leur RATIO : ici ~17/11 = 1,55, ce qui
    traduit une dispersion des erreurs (quelques prédictions nettement moins bonnes).

    NOTE MÉTHODO : on renvoie les mêmes métriques pour le TRAIN et le TEST.
    Comparer les deux est le moyen de détecter un surapprentissage : un score
    excellent sur le train mais médiocre sur le test = le modèle a mémorisé.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# =============================================================================
# 5. PROGRAMME PRINCIPAL
# =============================================================================

def main():
    # --- Options de la ligne de commande -------------------------------------
    # argparse permet de changer les réglages SANS éditer le code :
    #   python train.py --model rf --n-estimators 300
    # Chaque option a une valeur par défaut, donc `python train.py` seul marche.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["linear", "rf", "both"], default="both")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--experiment", default="getaround-pricing",
                        help="Nom de l'expérience MLflow. En changer force une "
                             "nouvelle création, donc une nouvelle artifact_location "
                             "(indispensable pour basculer vers S3).")
    args = parser.parse_args()

    # --- Configuration du suivi MLflow ---------------------------------------
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    # On affiche le backend retenu : très utile pour vérifier d'un coup d'œil
    # si on logge en local ou dans la base distante.
    backend = "base distante" if uri.startswith(("postgresql", "sqlite", "http")) else "local (mlruns/)"
    print(f"MLflow tracking -> {backend}")
    setup_experiment(args.experiment)     # au lieu de setup_experiment("getaround-pricing")

    # --- Données -------------------------------------------------------------
    df = load_and_clean()
    X = df[NUMERIC + CATEGORICAL + BOOLEAN]   # les variables explicatives
    y = df[TARGET]                            # la cible à prédire

    # NOTE : on entraîne sur la cible BRUTE (en euros), sans transformation log.
    # Ce choix a été TESTÉ, pas supposé : la distribution des prix est quasi
    # symétrique (moyenne 121 ~ médiane 119, skewness 0,61), et entraîner sur
    # log(prix) DÉGRADE les métriques (RMSE 17,7 vs 17,0). Le log corrige une
    # forte asymétrie ; il n'y en a pas ici, donc il déforme sans rien réparer.

    # Découpage train / test :
    #   - le TEST est mis de côté et ne sert QUE pour l'évaluation finale, sur
    #     des voitures que le modèle n'a jamais vues -> mesure honnête ;
    #   - random_state=42 fige le tirage : le même découpage à chaque exécution,
    #     donc des scores comparables entre deux entraînements.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # --- Entraînement des modèles --------------------------------------------
    models = ["linear", "rf"] if args.model == "both" else [args.model]

    # On garde trace du meilleur modèle au fil de la boucle. On l'initialise
    # avec un RMSE infini pour que le tout premier modèle testé le batte
    # forcément (astuce classique de recherche de minimum).
    best = {"rmse": np.inf, "pipe": None, "name": None}

    for name in models:
        # `with mlflow.start_run(...)` ouvre un RUN : tout ce qui est loggé à
        # l'intérieur du bloc y est rattaché, et le run est proprement fermé à
        # la sortie du bloc (même en cas d'erreur). Un run = un entraînement.
        
        # Nom du run : par défaut le nom du modèle ("linear" / "rf"). Lors d'un
        # BALAYAGE d'hyperparamètres, plusieurs runs partagent le même `name` et
        # deviennent indiscernables dans l'interface -> on suffixe la forêt par
        # sa profondeur ("rf-d4", "rf-d8"...). La régression linéaire n'a pas
        # d'hyperparamètre, elle garde son nom simple.
        run_name = f"{name}-d{args.max_depth}" if name == "rf" else name
        with mlflow.start_run(run_name=name):

            # Le PIPELINE : prétraitement puis modèle, en série. L'appel à
            # .fit() enchaîne tout automatiquement, dans le bon ordre.
            pipe = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("model", get_model(name, args.n_estimators, args.max_depth)),
            ])
            pipe.fit(X_train, y_train)

            # Évaluation sur les deux jeux : l'écart train/test révèle un
            # éventuel surapprentissage.
            train_m = evaluate(y_train, pipe.predict(X_train))
            test_m = evaluate(y_test, pipe.predict(X_test))

            # --- Traçage MLflow ---
            # log_param  : une valeur de CONFIGURATION (fixée avant l'entraînement)
            # log_metric : une valeur MESURÉE (résultat de l'entraînement)
            # log_model  : le modèle lui-même (part dans l'artifact store)
            mlflow.log_param("model", name)
            mlflow.log_param("n_rows", len(df))
            if name == "rf":   # ces hyperparamètres n'existent que pour la forêt
                mlflow.log_param("n_estimators", args.n_estimators)
                mlflow.log_param("max_depth", args.max_depth)

            # On préfixe par train_/test_ pour pouvoir comparer les deux
            # dans l'interface MLflow.
            for k, v in train_m.items():
                mlflow.log_metric(f"train_{k}", v)
            for k, v in test_m.items():
                mlflow.log_metric(f"test_{k}", v)

            mlflow.sklearn.log_model(pipe, name="model")

            print(f"\n[{name}]  test RMSE={test_m['rmse']:.2f}  "
                  f"MAE={test_m['mae']:.2f}  R²={test_m['r2']:.3f}")

            # Sélection du meilleur : on compare sur le RMSE de TEST (jamais du
            # train, qui favoriserait le modèle le plus surappris).
            if test_m["rmse"] < best["rmse"]:
                best = {"rmse": test_m["rmse"], "pipe": pipe, "name": name}

    # --- Sérialisation du meilleur modèle pour l'API (Partie 3) --------------
    # mkdir(exist_ok=True) : crée le dossier s'il n'existe pas, sans erreur s'il
    # existe déjà.
    ARTIFACT_DIR.mkdir(exist_ok=True)
    out = ARTIFACT_DIR / "model.joblib"

    # joblib.dump = SÉRIALISATION : on fige l'objet Python (ici le pipeline
    # ENTIER, prétraitement inclus) dans un fichier réutilisable sans
    # réentraînement. L'API fera l'opération inverse avec joblib.load().
    # compress=3 : compression -> ~2 Mo au lieu de ~62 Mo, plus simple à
    # versionner et à déployer.
    joblib.dump(best["pipe"], out, compress=3)

    print(f"\n✅ Meilleur modèle : {best['name']} (RMSE={best['rmse']:.2f})")
    print(f"   Sérialisé dans : {out}")


# Cette condition signifie : "n'exécute main() que si ce fichier est lancé
# DIRECTEMENT (python train.py)". Si un autre script fait `import train` pour
# réutiliser une fonction (comme le fait le notebook), main() ne se déclenche pas.
if __name__ == "__main__":
    main()

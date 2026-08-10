"""
=============================================================================
 GetAround — API de prédiction du prix de location (Partie 3)
=============================================================================

RÔLE DE CE FICHIER
------------------
Mettre le modèle entraîné en Partie 2 À DISPOSITION d'autres programmes, via
le réseau. C'est l'objectif "donner accès à échelle aux prédictions à
l'ensemble des équipes métier" du bloc 5. On dit qu'on **sert** le modèle
(en anglais : *model serving*) — comme un serveur sert un plat : l'API met le
modèle à disposition du client qui le demande.

ENDPOINTS EXPOSÉS
-----------------
  GET  /         -> page d'accueil + lien vers la doc
  GET  /health   -> état du service (vivant ? modèle chargé ?)
  POST /predict  -> reçoit les caractéristiques d'une voiture, renvoie le prix
  GET  /docs     -> documentation interactive, générée AUTOMATIQUEMENT par FastAPI

MODÈLE CLIENT / SERVEUR (rappel du vocabulaire)
-----------------------------------------------
Le CLIENT (un curl, un navigateur, un script Python) envoie une REQUÊTE HTTP ;
le SERVEUR (cette API) renvoie une RÉPONSE, ici en JSON. Le verbe HTTP indique
l'intention : GET = "donne-moi une information", POST = "voici des données,
traite-les et réponds".

⚠️ OÙ VIT LE MODÈLE QUI SERT RÉELLEMENT (point d'architecture clé)
------------------------------------------------------------------
Le fichier model.joblib existe en TROIS COPIES INDÉPENDANTES :
  1. dans le conteneur Docker sur Hugging Face -> LA SEULE qui sert les
     requêtes (chargée en mémoire au démarrage) ;
  2. sur GitHub -> une archive du code source ; GitHub n'exécute rien ;
  3. en local -> la copie de travail.
CONSÉQUENCE : modifier le modèle sur GitHub ne change RIEN à l'API en ligne.
Pour mettre à jour le modèle servi, il faut REDÉPLOYER le Space (reconstruction
du conteneur) — ou, en architecture plus avancée, passer par un Model Registry
auquel l'API demanderait "la version Production".
=============================================================================
"""
from pathlib import Path      # chemins portables (Windows / Linux / conteneur)

import joblib                 # désérialisation : relire le modèle depuis le disque
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field   # validation et documentation des données


# =============================================================================
# 1. CHARGEMENT DU MODÈLE
# =============================================================================

# Path(__file__).parent = le dossier de CE fichier. Le modèle est cherché à côté
# de app.py, donc le chemin reste valide où que l'app soit lancée (en local
# comme dans l'image Docker, où tout est copié dans /app).
MODEL_PATH = Path(__file__).parent / "model.joblib"

# ⚠️ LIGNE ESSENTIELLE, ET SA POSITION EST VOLONTAIRE.
# Elle est écrite AU NIVEAU DU MODULE (en dehors de toute fonction), donc elle
# s'exécute UNE SEULE FOIS, au démarrage du serveur. Le modèle est alors chargé
# en MÉMOIRE VIVE et y reste : chaque requête /predict réutilise cet objet déjà
# prêt. Résultat : réponse quasi instantanée.
#
# Si on avait mis ce joblib.load() À L'INTÉRIEUR de la fonction predict(), on
# relirait le fichier disque (~2 Mo) et on reconstruirait le modèle À CHAQUE
# REQUÊTE -> lenteur inutile.
#
# C'est ici que la SÉRIALISATION prend tout son sens : le modèle a été entraîné
# une fois (Partie 2), figé dans un fichier, et il est simplement "ressuscité"
# ici. L'API ne réentraîne JAMAIS.
model = joblib.load(MODEL_PATH)


# =============================================================================
# 2. CRÉATION DE L'APPLICATION
# =============================================================================

# L'objet FastAPI représente l'application. Les métadonnées passées ici
# (title, description, version) ne sont pas décoratives : elles alimentent
# directement la documentation interactive générée sur /docs. C'est ce qui
# permet de livrer une API "documentée" sans écrire une ligne de doc.
app = FastAPI(
    title="GetAround — Car Rental Price Prediction API",
    description="Prédit le prix de location journalier optimal d'une voiture "
                "à partir de ses caractéristiques. Voir **/docs** pour tester.",
    version="1.0.0",
)


# =============================================================================
# 3. SCHÉMAS DE DONNÉES (Pydantic)
# =============================================================================

class CarFeatures(BaseModel):
    """
    Schéma d'ENTRÉE : décrit les caractéristiques attendues pour une voiture.

    À QUOI ÇA SERT — Pydantic remplit trois rôles d'un coup :

      1. VALIDATION. Chaque champ est typé (str, int, bool). Si la requête
         arrive avec un champ manquant ou du texte là où un nombre est attendu,
         FastAPI répond automatiquement une erreur HTTP 422 ("Unprocessable
         Entity") détaillant le problème — AVANT même que le modèle soit
         sollicité. L'API ne plante donc pas sur une requête mal formée : elle
         explique poliment ce qui cloche.

      2. DOCUMENTATION. Ce schéma est lu par FastAPI pour construire la page
         /docs : les noms de champs, leurs types et les exemples ci-dessous y
         apparaissent, et le bouton "Try it out" est pré-rempli avec eux.

      3. CONVERSION. Les données JSON reçues sont transformées en un objet
         Python propre (`car`), avec les bons types.

    SYNTAXE — `Field(..., examples=[...])` :
      - les trois points `...` (l'objet Ellipsis) signifient "champ OBLIGATOIRE".
        Sans valeur par défaut, une requête qui l'omet est rejetée ;
      - `examples` fournit la valeur d'exemple affichée dans /docs.

    ⚠️ Les noms de ces champs doivent correspondre EXACTEMENT aux noms des
    colonnes utilisées à l'entraînement : le pipeline les retrouve par leur nom.
    """
    model_key: str = Field(..., examples=["Citroën"])
    mileage: int = Field(..., examples=[140000])
    engine_power: int = Field(..., examples=[100])
    fuel: str = Field(..., examples=["diesel"])
    paint_color: str = Field(..., examples=["black"])
    car_type: str = Field(..., examples=["sedan"])
    private_parking_available: bool = Field(..., examples=[True])
    has_gps: bool = Field(..., examples=[True])
    has_air_conditioning: bool = Field(..., examples=[False])
    automatic_car: bool = Field(..., examples=[False])
    has_getaround_connect: bool = Field(..., examples=[True])
    has_speed_regulator: bool = Field(..., examples=[True])
    winter_tires: bool = Field(..., examples=[False])


class PredictionOut(BaseModel):
    """
    Schéma de SORTIE : décrit la forme de la réponse.

    Déclaré pour deux raisons :
      - il DOCUMENTE la réponse dans /docs (le lecteur sait à quoi s'attendre) ;
      - associé à `response_model=` sur l'endpoint, il garantit le format
        renvoyé (contrat d'interface stable pour les équipes qui consomment l'API).
    """
    rental_price_per_day: float


# =============================================================================
# 4. ENDPOINTS
# =============================================================================
# Le "décorateur" @app.get(...) / @app.post(...) placé au-dessus d'une fonction
# dit à FastAPI : "quand une requête arrive sur cette route avec ce verbe HTTP,
# exécute cette fonction". La valeur retournée (un dictionnaire Python) est
# automatiquement convertie en JSON dans la réponse.

@app.get("/")
def home():
    """
    Page d'accueil : oriente vers la documentation interactive.

    Utile en pratique : sans cet endpoint, ouvrir l'URL racine du Space
    afficherait une erreur 404, ce qui est déroutant pour quelqu'un qui découvre
    l'API (un examinateur, par exemple). Ici, il tombe sur un message clair
    et sait où aller.
    """
    return {
        "message": "GetAround car rental price prediction API.",
        "documentation": "/docs",
        "predict_endpoint": "POST /predict",
    }


@app.get("/health")
def health():
    """
    Vérifie que le service tourne et que le modèle est bien chargé.

    Un endpoint de "santé" (health check) est un STANDARD DE PRODUCTION : les
    hébergeurs et les outils de supervision l'interrogent régulièrement pour
    savoir si le service répond encore. S'il ne répond plus, on peut alerter
    ou redémarrer automatiquement le conteneur.

    Ici on vérifie deux choses d'un coup : que l'API répond (elle renvoie
    quelque chose) ET que le modèle a bien été chargé au démarrage.
    """
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOut)
def predict(car: CarFeatures):
    """
    Prédit le prix de location journalier d'une voiture.

    POURQUOI POST ET PAS GET : on ENVOIE des données (13 caractéristiques) dans
    le corps de la requête pour obtenir un résultat calculé. GET sert à demander
    une ressource, POST à soumettre des données à traiter.

    L'ANNOTATION `car: CarFeatures` EST LE DÉCLENCHEUR : c'est en voyant ce type
    que FastAPI comprend qu'il doit lire le corps JSON de la requête, le valider
    contre le schéma, et fournir ici un objet `car` propre. Rien à écrire de plus.
    """
    # model_dump() convertit l'objet Pydantic en dictionnaire Python simple.
    # On l'enveloppe dans une liste pour créer un DataFrame D'UNE SEULE LIGNE :
    # le pipeline scikit-learn attend un tableau à 2 dimensions (des lignes et
    # des colonnes), même pour une seule observation.
    #
    # ⚠️ POINT CLÉ — ON PASSE DES FEATURES BRUTES ("Citroën", "diesel", 140000)
    # sans les encoder nous-mêmes. C'est possible parce qu'en Partie 2 on a
    # sérialisé le PIPELINE COMPLET : le StandardScaler et le OneHotEncoder sont
    # embarqués dedans et s'appliquent automatiquement, exactement comme à
    # l'entraînement. Si on n'avait sauvegardé que le RandomForest, il faudrait
    # reproduire ici toute la logique d'encodage — avec le risque qu'elle diverge
    # de celle de l'entraînement (ce qu'on appelle le "train/serving skew").
    X = pd.DataFrame([car.model_dump()])

    # .predict(X) renvoie un TABLEAU de prédictions (une par ligne d'entrée).
    # Comme on n'a qu'une ligne, on prend l'élément [0].
    # float(...) convertit le type numpy en type Python natif : indispensable,
    # car un np.float64 n'est pas directement convertible en JSON.
    prediction = float(model.predict(X)[0])

    # round(..., 2) : deux décimales suffisent pour un prix en euros.
    # Le dictionnaire renvoyé correspond au schéma PredictionOut déclaré plus
    # haut, et FastAPI le sérialise en JSON : {"rental_price_per_day": 115.6}
    return {"rental_price_per_day": round(prediction, 2)}

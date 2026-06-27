"""
=============================================================================
 GetAround — API de prédiction du prix de location (Partie 3)
=============================================================================
API FastAPI exposant le modèle entraîné en Partie 2.

Endpoints :
  GET  /         -> page d'accueil + lien vers la doc
  GET  /health   -> état du service (vivant ? modèle chargé ?)
  POST /predict  -> reçoit les caractéristiques d'une voiture, renvoie le prix

La doc interactive est générée AUTOMATIQUEMENT par FastAPI sur /docs.
Le modèle (pipeline complet : prétraitement + RandomForest) est chargé UNE
FOIS au démarrage depuis model.joblib — pas de réentraînement par requête.
=============================================================================
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- Chargement du modèle au démarrage (sérialisation -> mise en production) ---
MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="GetAround — Car Rental Price Prediction API",
    description="Prédit le prix de location journalier optimal d'une voiture "
                "à partir de ses caractéristiques. Voir **/docs** pour tester.",
    version="1.0.0",
)


# --- Schéma d'ENTRÉE : valide et documente les caractéristiques attendues ---
# Pydantic rejette automatiquement une requête mal formée (champ manquant,
# mauvais type) avec un message d'erreur clair, avant même d'atteindre le modèle.
class CarFeatures(BaseModel):
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


# --- Schéma de SORTIE (documente la réponse) ---
class PredictionOut(BaseModel):
    rental_price_per_day: float


@app.get("/")
def home():
    """Page d'accueil : oriente vers la documentation interactive."""
    return {
        "message": "GetAround car rental price prediction API.",
        "documentation": "/docs",
        "predict_endpoint": "POST /predict",
    }


@app.get("/health")
def health():
    """Vérifie que le service tourne et que le modèle est bien chargé."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOut)
def predict(car: CarFeatures):
    """
    Prédit le prix de location journalier d'une voiture.

    Le pipeline embarqué applique tout le prétraitement (encodage,
    standardisation) puis le modèle — on lui passe donc des features BRUTES.
    """
    # 1 ligne de DataFrame avec les mêmes noms de colonnes qu'à l'entraînement.
    X = pd.DataFrame([car.model_dump()])
    prediction = float(model.predict(X)[0])
    return {"rental_price_per_day": round(prediction, 2)}

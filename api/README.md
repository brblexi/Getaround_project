---
title: GetAround Pricing API
emoji: 🚗
colorFrom: purple
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# GetAround — Car Rental Price Prediction API

API **FastAPI** qui prédit le prix de location journalier optimal d'une voiture
à partir de ses caractéristiques. Le modèle (pipeline scikit-learn complet) est
entraîné en Partie 2 et embarqué ici sous `model.joblib`.

## Endpoints

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/` | message d'accueil + lien vers la doc |
| GET | `/health` | état du service (modèle chargé ?) |
| POST | `/predict` | prédit le prix à partir des caractéristiques |
| GET | `/docs` | **documentation interactive** (générée par FastAPI) |

## Tester avec `curl`

```bash
curl -X POST "https://<votre-space>.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_key": "Citroën",
    "mileage": 140000,
    "engine_power": 100,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "sedan",
    "private_parking_available": true,
    "has_gps": true,
    "has_air_conditioning": false,
    "automatic_car": false,
    "has_getaround_connect": true,
    "has_speed_regulator": true,
    "winter_tires": false
  }'
```

Réponse :

```json
{"rental_price_per_day": 115.6}
```

## Lancer en local

```bash
pip install -r requirements.txt
uvicorn app:app --reload          # http://localhost:8000/docs
```

Ou avec Docker (comme sur le Space) :

```bash
docker build -t getaround-api .
docker run -p 7860:7860 getaround-api
# http://localhost:7860/docs
```

## Déployer sur Hugging Face Spaces

1. Créer un Space → **SDK : Docker**.
2. Pousser le contenu de ce dossier `api/` (ce README avec son entête YAML doit
   être à la racine du Space, et `model.joblib` doit être présent).
3. Le Space construit le Dockerfile et sert l'API sur le port 7860.

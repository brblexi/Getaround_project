# GetAround — Projet bloc 5 (Jedha CDSD)

Projet de fin de bloc 5 (industrialisation et déploiement d'un modèle de ML).
À partir des données GetAround, deux volets :

- l'analyse de l'impact des retours tardifs sur la location suivante, restituée
  dans un dashboard interactif ;
- un modèle de prédiction du prix de location journalier, exposé via une API.

## Liens

- Dépôt GitHub : https://github.com/brblexi/getaround_project
- Dashboard (analyse des délais) : https://huggingface.co/spaces/AlexBarbier/getaround-dashboard
- API de prédiction de prix : https://huggingface.co/spaces/AlexBarbier/getaround-api

## Organisation du dépôt

```
getaround/
├── notebooks/      EDA — exploration des deux jeux de données
├── dashboard/      Partie 1 — dashboard Streamlit (analyse des délais)
├── model/          Partie 2 — entraînement du modèle + suivi MLflow
└── api/            Partie 3 — API FastAPI (endpoint /predict)
```

Chaque dossier déployable (`dashboard/`, `api/`) a son propre `requirements.txt`
et son `Dockerfile`.

## Partie 1 — Analyse des délais

Le dashboard quantifie la fréquence des retours en retard et leur impact sur la
location suivante (le taux d'annulation passe d'environ 11 % à 17 % quand la
location est impactée par un retard précédent), puis propose un simulateur du
seuil de délai minimum entre deux locations et de son périmètre (toutes les
voitures ou Connect uniquement), pour arbitrer entre problèmes résolus et
locations bloquées.

Pour lancer en local :

```
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Partie 2 — Modèle de pricing

Nettoyage des données, pipeline scikit-learn (encodage + standardisation),
comparaison d'une régression linéaire et d'un RandomForest, suivi des
expériences avec MLflow. Le meilleur modèle (RandomForest, RMSE ≈ 17 €,
R² ≈ 0,75) est sérialisé dans `model/artifacts/model.joblib` et réutilisé par
l'API.

```
cd model
pip install -r requirements.txt
python train.py        # entraînement + log MLflow
mlflow ui              # http://localhost:5000
```

## Partie 3 — API de prédiction

API FastAPI qui charge le modèle et expose l'endpoint `POST /predict`. La
documentation interactive est disponible sur `/docs`.

Exemple avec curl :

```
curl -X POST "https://AlexBarbier-getaround-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"model_key":"Citroën","mileage":140000,"engine_power":100,"fuel":"diesel","paint_color":"black","car_type":"sedan","private_parking_available":true,"has_gps":true,"has_air_conditioning":false,"automatic_car":false,"has_getaround_connect":true,"has_speed_regulator":true,"winter_tires":false}'
```

Réponse :

```
{"rental_price_per_day": 115.6}
```

Exemple en Python :

```python
import requests

car = {
    "model_key": "Citroën", "mileage": 140000, "engine_power": 100,
    "fuel": "diesel", "paint_color": "black", "car_type": "sedan",
    "private_parking_available": True, "has_gps": True,
    "has_air_conditioning": False, "automatic_car": False,
    "has_getaround_connect": True, "has_speed_regulator": True,
    "winter_tires": False,
}
r = requests.post("https://AlexBarbier-getaround-api.hf.space/predict", json=car)
print(r.json())
```

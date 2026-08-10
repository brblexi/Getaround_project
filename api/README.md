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

API **FastAPI** qui prédit le prix de location journalier optimal d'une voiture à
partir de ses caractéristiques.

Le modèle est un pipeline scikit-learn complet, entraîné par `model/train.py` et
embarqué ici sous forme de fichier `model.joblib`. L'API se contente de le charger
au démarrage et de l'exposer derrière une route HTTP : aucun entraînement n'a lieu
dans ce service.

**URL de production :** https://alexbarbier-getaround-api.hf.space

## Endpoints

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/` | message d'accueil + lien vers la doc |
| GET | `/health` | état du service (modèle chargé ?) |
| POST | `/predict` | prédit le prix à partir des caractéristiques |
| GET | `/docs` | **documentation interactive** (générée par FastAPI) |

## Contenu du dossier

```
api/
├── app.py           # définition FastAPI (routes + schéma Pydantic)
├── model.joblib     # pipeline scikit-learn sérialisé
├── requirements.txt # dépendances Python
├── Dockerfile       # environnement d'exécution
├── .dockerignore    # fichiers exclus du contexte de build
└── README.md        # ce fichier (son entête YAML configure le Space)
```

Ce dossier est **autonome** : il ne dépend d'aucun fichier des dossiers `model/`
ou `dashboard/`. C'est ce qui permet de le pousser tel quel vers un Space.

---

# Utilisation

Trois façons de faire tourner cette API, de la plus légère à la plus proche de la
production. Elles ne s'excluent pas : les étapes 1 et 2 servent à valider avant de
déployer à l'étape 3.

## 1. En local, sans Docker

Le plus rapide pour développer. Nécessite Python 3.11 et un environnement virtuel
activé.

```bash
cd api/
pip install -r requirements.txt
uvicorn app:app --reload
```

L'API écoute sur **http://localhost:8000** — documentation interactive sur
http://localhost:8000/docs.

> **`uvicorn app:app`** — le premier `app` est le module (`app.py`), le second est
> la variable `app = FastAPI()` définie dedans.
>
> **`--reload`** redémarre le serveur à chaque modification de fichier. Pratique en
> développement, **à ne jamais utiliser en production** (surcoût mémoire et
> comportement imprévisible sous charge).

## 2. En local, avec Docker

Reproduit exactement l'environnement du Space. C'est la répétition générale : si
cette étape passe, le déploiement passera.

```bash
cd api/
docker build -t getaround-api .
docker run -p 7860:7860 getaround-api
```

L'API écoute sur **http://localhost:7860** — doc sur http://localhost:7860/docs.

> **Pourquoi 7860 ici et 8000 juste au-dessus ?** Hugging Face Spaces attend que
> l'application écoute sur le port déclaré dans `app_port` de l'entête YAML — 7860
> par convention. Le `Dockerfile` fige donc ce port dans sa commande de démarrage :
>
> ```dockerfile
> CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
> ```
>
> C'est la même commande qu'à l'étape 1, sans `--reload`, et avec `--host 0.0.0.0`
> pour que le serveur accepte les connexions venant de l'extérieur du conteneur
> (par défaut uvicorn n'écoute que sur `127.0.0.1`, donc le proxy de Hugging Face
> ne pourrait pas le joindre).
>
> Le `-p 7860:7860` mappe ce port du conteneur vers le même port de ta machine.

## 3. Déployer sur Hugging Face Spaces

1. **Créer un Space → SDK : `Docker`.**

   Le « SDK » indique à HF comment exécuter l'application. Les modes Streamlit,
   Gradio ou Static sont des raccourcis pour des cas standards. Choisir `Docker`
   signifie « ne présume rien, lis mon Dockerfile et suis-le à la lettre » — c'est
   ce qu'on veut ici, puisque le Dockerfile décrit déjà précisément
   l'environnement voulu (Python 3.11, dépendances figées, uvicorn).

2. **Pousser le *contenu* du dossier `api/`** à la racine du Space — pas le dossier
   lui-même. Ce README et son entête YAML doivent se retrouver à la racine, et
   `model.joblib` doit être présent.

   ```bash
   git clone https://huggingface.co/spaces/AlexBarbier/getaround-api
   cd getaround-api

   git lfs install
   git lfs track "*.joblib"          # avant d'ajouter le modèle

   cp -r ../Projet_GetAround/api/. .

   git add -A
   git commit -m "Deploy GetAround pricing API"
   git push
   ```

   Sous PowerShell, la copie s'écrit :
   `Copy-Item -Path ..\Projet_GetAround\api\* -Destination . -Recurse -Force`

3. **Le Space construit l'image et démarre le conteneur.** Le statut passe en
   *Building* ; l'onglet **Logs** affiche la sortie du `docker build` en cas
   d'échec. Une fois *Running*, l'API est servie sur le port 7860 et accessible à
   l'adresse publique du Space.

---

# Tester l'API

La même requête fonctionne aux trois étapes ci-dessus : seule l'URL de base change.

| Étape | URL de base |
|---|---|
| 1. local sans Docker | `http://localhost:8000` |
| 2. local avec Docker | `http://localhost:7860` |
| 3. Space déployé | `https://alexbarbier-getaround-api.hf.space` |

Le plus simple pour un premier essai est d'ouvrir `/docs` dans un navigateur :
FastAPI y génère un formulaire permettant d'envoyer une requête sans écrire une
ligne de code.

## Avec `curl`

```Powershell
$body = @{
    model_key = "Citroen"
    mileage = 140000
    engine_power = 100
    fuel = "diesel"
    paint_color = "black"
    car_type = "sedan"
    private_parking_available = $true
    has_gps = $true
    has_air_conditioning = $false
    automatic_car = $false
    has_getaround_connect = $true
    has_speed_regulator = $true
    winter_tires = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://alexbarbier-getaround-api.hf.space/predict" -Method Post -Body $body -ContentType "application/json"
```


```bash
curl -X POST "https://alexbarbier-getaround-api.hf.space/predict" \
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

> Sous PowerShell, `curl` est un alias d'`Invoke-WebRequest` et n'accepte pas cette
> syntaxe. Utiliser `curl.exe` pour appeler le vrai client, en échappant les
> guillemets internes par des backslashes.

## Avec Python

```python
import requests

BASE_URL = "https://alexbarbier-getaround-api.hf.space"

payload = {
    "model_key": "Citroën",
    "mileage": 140000,
    "engine_power": 100,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "sedan",
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": False,
    "automatic_car": False,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": False,
}

response = requests.post(f"{BASE_URL}/predict", json=payload)
print(response.json())
```

## Vérifier que le service est vivant

```bash
curl https://alexbarbier-getaround-api.hf.space/health
```

Réponse attendue :

```json
{"status":"ok","model_loaded":true}
```

Le champ `model_loaded` confirme que `model.joblib` a bien été trouvé et
désérialisé au démarrage du conteneur. C'est la première chose à vérifier après un
déploiement.

---

# Limites connues

**Marques hors référentiel.** Le champ `model_key` n'est pas validé en entrée. Si
la valeur envoyée n'était pas présente dans les données d'entraînement (par exemple
`"Ferrari"`), ou par simple erreur de saisie (la correspondance est stricte, sensible
à la casse et aux accents), l'encodeur du pipeline la **neutralise** : la requête renvoie un code 200
et un prix, mais celui-ci est calculé **sans tenir compte de la marque**, à partir
des seules autres caractéristiques. La prédiction reste cohérente statistiquement,
mais n'est pas fiable pour un véhicule dont la marque est déterminante dans la
formation du prix.

**Pas d'intervalle de confiance.** L'API renvoie une valeur ponctuelle. Elle ne
quantifie pas l'incertitude associée, alors que celle-ci varie selon la densité des
données d'entraînement autour du profil demandé.

**Modèle figé dans l'image.** `model.joblib` est copié dans l'image Docker au build.
Toute mise à jour du modèle impose de reconstruire et redéployer le Space ; il n'y a
ni rechargement à chaud, ni versionnage des prédictions.

**Pas d'authentification ni de quota.** L'endpoint `/predict` est public et sans
limitation de débit. Acceptable pour une démonstration, à ne pas conserver tel quel
en production.

**Conteneur sans état.** Le Space ne persiste rien entre deux redémarrages. Sans
conséquence ici puisque l'API se contente de lire le modèle, mais tout ajout
d'écriture de fichiers serait perdu.

# Évolutions possibles

- **Valider `model_key` à l'entrée** : restreindre le champ aux catégories
  effectivement vues à l'entraînement, extraites du pipeline chargé plutôt que
  recopiées à la main. L'API renverrait alors un 422 explicite plutôt qu'un prix
  silencieusement approximatif, et un endpoint `GET /model-keys` exposerait le
  référentiel des valeurs acceptées.
- **Renvoyer un intervalle de prédiction** plutôt qu'une valeur ponctuelle, par
  exemple à partir de la dispersion des prédictions individuelles des arbres.
- **Journaliser les requêtes entrantes** pour suivre la dérive des données par
  rapport à la distribution d'entraînement, et déclencher un réentraînement quand
  l'écart devient significatif.
- **Protéger l'accès** par clé d'API et appliquer une limitation de débit.
- **Découpler le modèle de l'image** en le chargeant depuis le Hub au démarrage,
  afin de mettre à jour le modèle sans redéployer l'API.

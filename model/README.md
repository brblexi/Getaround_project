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
├── .env.example                    # variables de configuration à recopier en .env
└── requirements.txt
```

## Démarche

Nettoyage (3 lignes aberrantes retirées) → pipeline scikit-learn
(`StandardScaler` + `OneHotEncoder` + booléens) → entraînement de
`LinearRegression` (baseline) vs `RandomForestRegressor` → suivi MLflow →
sérialisation du meilleur modèle.

**Résultats (jeu de test) :** le RandomForest l'emporte avec RMSE ≈ 17 €,
MAE ≈ 11 €, R² ≈ 0,75 (baseline linéaire : RMSE ≈ 18 €, R² ≈ 0,70).

## Installer les dépendances

Une seule fois, dans le venv du projet :

```bash
pip install -r requirements.txt
```

## Lancer l'entraînement

`python train.py` suffit : chaque option a une valeur par défaut. Les lignes
suivantes sont des **variantes**, pas une séquence — on en lance une seule, et
les options se combinent.

```bash
python train.py                                     # 2 modèles, réglages par défaut
python train.py --model rf                          # un seul modèle (linear | rf | both)
python train.py --n-estimators 300 --max-depth 20   # autres hyperparamètres de la forêt
python train.py --experiment autre-nom              # autre expérience MLflow
python train.py --test-size 0.3                     # autre proportion train/test
```

Passer par des options plutôt que par l'édition du code rend chaque entraînement
reproductible : MLflow enregistre les paramètres effectifs du run.


## Suivi des expériences : où atterrissent les données

Le suivi **bascule automatiquement** selon les variables d'environnement
(`get_tracking_uri()` + `setup_experiment()`), avec une séparation
**backend store / artifact store** :

| Donnée | Va dans | Variable |
|---|---|---|
| Métriques, paramètres, métadonnées des runs | **PostgreSQL / Neon** | `MLFLOW_TRACKING_URI` |
| Modèle sérialisé, fichiers loggés | **S3** | `MLFLOW_ARTIFACT_LOCATION` (+ clés AWS) |

`train.py` charge le fichier `.env` au démarrage (`load_dotenv`) : il n'y a donc
**rien à exporter à la main** avant de lancer un entraînement. Si le `.env` est
absent — ou si les variables n'y sont pas renseignées — le script retombe
automatiquement sur un stockage **local** (`mlruns/`).

> 🏭 **Pourquoi `load_dotenv` et pas une lecture de fichier de configuration ?**
> Le code ne lit que `os.environ`, jamais le `.env` directement. C'est le
> contrat de production : sur une plateforme (Docker, Space Hugging Face, ECS),
> les variables sont **injectées par l'hébergeur** et aucun `.env` n'existe —
> l'appel ne trouve rien et ne fait rien. Le `.env` n'est qu'une commodité de
> développement local, avec `override=False` pour qu'une variable déjà définie
> dans l'environnement ait toujours le dernier mot.

### Prérequis d'infrastructure

À provisionner une fois, avant le premier entraînement distant :

1. **Base PostgreSQL dédiée** (Neon) — ne pas réutiliser une base contenant déjà
   des tables MLflow d'un autre projet : les expériences et les modèles
   enregistrés se mélangeraient. Utiliser le point de terminaison **direct**
   (sans `-pooler`) : MLflow crée son schéma par migrations Alembic à la
   première connexion, ce que le pooler en mode transaction supporte mal.
2. **Bucket S3** — accès public bloqué, versioning désactivé (c'est MLflow qui
   versionne, un dossier par run). Sa région est **figée à la création** et doit
   correspondre exactement à `AWS_DEFAULT_REGION`.
3. **Utilisateur IAM dédié** avec une politique restreinte à ce seul bucket
   (`ListBucket` + `GetBucketLocation` sur le bucket, `PutObject` / `GetObject` /
   `DeleteObject` sur `bucket/*`), plutôt que `AmazonS3FullAccess` — les clés
   vivent dans un fichier local, autant limiter la portée d'une fuite.

### Configuration

Copier `.env.example` en `.env` et le remplir. Puis :

```powershell
# PowerShell
python train.py    # métriques -> Neon, artefacts -> S3
```

```bash
# bash / Linux
python train.py
```

Deux lignes de confirmation s'affichent au démarrage :

```
MLflow tracking -> base distante
Artefacts -> s3://<bucket>/mlflow-artifacts
```

La seconde n'apparaît **qu'à la création de l'expérience**. Son absence signifie
que l'expérience existait déjà — voir l'encadré ci-dessous.

### Visualiser les expériences

**En local** (aucune variable définie) :

```bash
mlflow ui      # http://localhost:5000
```

**Branché sur Neon.** `mlflow ui` est un **processus séparé** qui ne lit pas le
`.env` : il faut lui passer explicitement l'URI.

```powershell
# PowerShell
$env:MLFLOW_TRACKING_URI = ((Get-Content .env | Select-String "^MLFLOW_TRACKING_URI=") -split "=", 2)[1]
mlflow ui --backend-store-uri $env:MLFLOW_TRACKING_URI
```

```bash
# bash / Linux
export $(grep -v '^#' .env | xargs)
mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"
```

Pour consulter aussi l'onglet *Artifacts* d'un run, le serveur a besoin des
identifiants AWS (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_DEFAULT_REGION`) dans son propre environnement.

> ⚠️ **Piège : `artifact_location` est figé à la création de l'expérience.**
>
> L'emplacement des artefacts est une **colonne de la table `experiments`**,
> écrite une seule fois à l'`INSERT` et jamais recalculée. Définir
> `MLFLOW_ARTIFACT_LOCATION` **après** la création d'une expérience n'a donc
> aucun effet sur elle — **et sans le moindre message d'erreur**.
>
> Corollaire : la variable d'environnement ne s'applique pas d'elle-même, il
> faut qu'elle soit **transmise** à `create_experiment(name, artifact_location=…)`.
> C'est ce que fait `setup_experiment()`. L'expérience `Default`, créée en
> interne par MLflow sans ce paramètre, en donne la contre-épreuve : elle
> conserve un emplacement local alors que la variable S3 est définie.
>
> Vérification avant tout entraînement :
> ```python
> print(mlflow.get_experiment_by_name("getaround-pricing").artifact_location)
> ```
> Si le résultat commence par `file:///`, les artefacts restent en local. Il
> faut alors supprimer l'expérience, ou en utiliser une neuve
> (`python train.py --experiment getaround-pricing-s3`).

> 🔐 **Secrets** : l'URL Neon (qui contient un mot de passe) et les clés AWS
> vivent dans `.env`, non versionné (voir `.gitignore`) — jamais en dur dans le
> code ni sur GitHub. Ils ne concernent que l'entraînement : l'API déployée sur
> Hugging Face ne fait aucun tracking MLflow, elle charge un `.joblib` et prédit.

> 📦 **Pourquoi deux stockages ?** Une base SQL (Neon) est faite pour la donnée
> structurée requêtable (chiffres, texte), pas pour des fichiers binaires ; S3
> est fait pour les fichiers. Neon conserve, pour chaque run, l'URI S3 de ses
> artefacts (colonne `artifact_uri` de la table `runs`) pour faire le lien.

> Le pipeline est sérialisé **deux fois, indépendamment**, par deux appels que
> rien ne relie dans `train.py` : `mlflow.sklearn.log_model()` l'envoie vers
> **S3**, et `joblib.dump()` écrit `artifacts/model.joblib` sur le disque.
> Deux usages distincts : l'artefact S3 sert la **traçabilité** (il est
> accompagné de `MLmodel`, `requirements.txt`, `python_env.yaml`, donc
> rechargeable sur une autre machine avec son environnement) ; le `.joblib`
> sert le **déploiement**, c'est lui qu'on copie dans l'image Docker de l'API.

> L'API `/predict` (Partie 3) recharge le .joblib tel quel et lui passe des 
> caractéristiques **brutes** — tout le prétraitement est embarqué avec le 
> modèle -> il s'entraîne avec le modèle, il se sérialise avec le modèle, 
> il se déploie avec le modèle. Le séparer, c'est créer deux artefacts qui 
> doivent rester synchronisés manuellement — et qui ne le resteront pas.

## Évolutions

- **Model Registry** — immédiatement accessible, maintenant que le backend est
  une base SQL (il ne fonctionne pas avec le file store local). Il ne manque
  qu'un `mlflow.register_model()` et une politique de promotion
  Staging → Production. L'API demanderait alors « la version Production » au
  registre au lieu d'embarquer un fichier figé dans son image.
  Ici on a préféré le modèle embarqué puisque: contrainte de démo jury, pas de 
  réentrainement fréquent, un seul service consomme le modèle. 
- **Monitoring de production** — d'une autre nature : il ne s'agit plus de
  comparer des entraînements mais de surveiller un modèle qui sert des requêtes
  réelles (dérive des données, dégradation des performances, latence). Cela
  suppose de **logger les prédictions de l'API**, ce qu'elle ne fait pas
  aujourd'hui, et de trouver un proxy métier pour mesurer la qualité — le
  « vrai » prix optimal n'étant jamais observé a posteriori.

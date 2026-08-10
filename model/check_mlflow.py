"""Vérification MLflow AVANT le premier entraînement.

Pourquoi un script séparé : artifact_location est écrit UNE FOIS À LA CRÉATION
de l'expérience et n'est plus modifiable ensuite. Lancer train.py directement
reviendrait à figer cette valeur sans l'avoir vue. Ici on regarde d'abord.
"""
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

uri = os.environ.get("MLFLOW_TRACKING_URI")
print(f"Tracking URI : {uri.split('@')[-1] if uri else 'NON DÉFINI (repli local)'}")
print(f"Artefacts    : {os.environ.get('MLFLOW_ARTIFACT_LOCATION', 'NON DÉFINI (local)')}")

mlflow.set_tracking_uri(uri)

# Premier vrai contact avec Neon : c'est CET appel qui crée les 34 tables
# du schéma MLflow si la base est vierge. Il peut prendre quelques secondes
# (Neon gratuit met le compute en veille après inactivité).
exps = mlflow.search_experiments()
print(f"\nConnexion Neon OK — {len(exps)} expérience(s) :")
for e in exps:
    print(f"  [{e.experiment_id}] {e.name}  ->  {e.artifact_location}")
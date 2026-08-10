"""Test de connexion S3 isolé — à lancer AVANT tout entraînement.

Objectif : valider séparément les briques (identifiants, région, permissions)
plutôt que de tout lancer d'un coup. Si `train.py` échoue plus tard, on saura
que le problème n'est PAS côté S3.
"""
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

# On extrait le nom du bucket depuis l'URI s3://bucket/prefixe :
# on retire le schéma, puis on coupe au premier "/".
bucket = os.environ["MLFLOW_ARTIFACT_LOCATION"].replace("s3://", "").split("/")[0]

s3 = boto3.client("s3")

# get_bucket_location interroge AWS sur la région RÉELLE du bucket.
# Curiosité historique : la réponse est None pour us-east-1, première région
# créée, qui fait donc figure d'exception.
region = s3.get_bucket_location(Bucket=bucket)["LocationConstraint"] or "us-east-1"
print(f"Bucket   : {bucket}")
print(f"Région   : {region}")
print(f"Dans .env: {os.environ.get('AWS_DEFAULT_REGION')}")

# Écriture puis suppression : valide PutObject et DeleteObject d'un coup.
s3.put_object(Bucket=bucket, Key="_test/ping.txt", Body=b"ok")
print("Écriture : OK")
s3.delete_object(Bucket=bucket, Key="_test/ping.txt")
print("Suppression : OK")
"""Standalone S3 connectivity check — run before any training.

Not a pytest file: this reaches out to real infrastructure and needs AWS
credentials, so it is a script you run deliberately rather than a test that
should pass anywhere. Named without the `test_` prefix for exactly that reason.

    python check_s3.py

Validating the pieces separately — credentials, region, permissions — means that
if `train.py` fails later, S3 is already ruled out.
"""

import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)


def main() -> int:
    artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION")
    if not artifact_location:
        print("MLFLOW_ARTIFACT_LOCATION is not set — nothing to check.")
        print("Fill in model/.env, or export it before running this script.")
        return 1

    # Extract the bucket from the s3://bucket/prefix URI: drop the scheme, then
    # cut at the first slash.
    bucket = artifact_location.replace("s3://", "").split("/")[0]
    s3 = boto3.client("s3")

    # get_bucket_location asks AWS for the bucket's real region. Historical
    # quirk: the answer is None for us-east-1, the first region ever created.
    region = s3.get_bucket_location(Bucket=bucket)["LocationConstraint"] or "us-east-1"
    print(f"Bucket      : {bucket}")
    print(f"Region      : {region}")
    print(f"In .env     : {os.environ.get('AWS_DEFAULT_REGION')}")

    # Write then delete: validates PutObject and DeleteObject in one go.
    s3.put_object(Bucket=bucket, Key="_test/ping.txt", Body=b"ok")
    print("Write       : OK")
    s3.delete_object(Bucket=bucket, Key="_test/ping.txt")
    print("Delete      : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
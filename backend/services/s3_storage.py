from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _client():
    try:
        import boto3  # type: ignore

        session = boto3.session.Session()
        return session.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
            region_name=os.getenv("S3_REGION") or None,
            aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID") or None,
            aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY") or None,
            config=boto3.session.Config(signature_version="s3v4"),  # type: ignore[attr-defined]
        )
    except Exception:
        return None


def upload_outputs(paths: Iterable[Path]) -> list[str]:
    """Upload given files to configured S3 bucket; return list of object keys.

    No-op if STORAGE_S3_ENABLED is not truthy or client/bucket are not configured.
    """
    if str(os.getenv("STORAGE_S3_ENABLED", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        return []
    bucket = os.getenv("S3_BUCKET_OUTPUT")
    if not bucket:
        return []
    cli = _client()
    if cli is None:
        return []
    uploaded: list[str] = []
    for p in paths:
        try:
            if not Path(p).exists():
                continue
            key = Path(p).name
            cli.upload_file(str(p), bucket, key)  # type: ignore[attr-defined]
            uploaded.append(key)
        except Exception:
            continue
    return uploaded

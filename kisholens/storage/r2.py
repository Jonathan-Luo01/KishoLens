"""
r2.py — Cloudflare R2 object storage integration for KishoLens.
Provides helper functions to sync pre-computed databases, caches, and embeddings
between Cloudflare R2 and local execution environments (such as Google Cloud Run).
"""

from __future__ import annotations
import os
from typing import Optional, List

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "kisholens-data")

DEFAULT_SYNC_FILES = [
    "stats_cache.sqlite",
    "stats_cache.json",
    "vector_cache.json",
    "arc_cache.json",
    "genre_centroids.npy",
    "genre_centroids_meta.json",
    "territory_centroids.npy",
    "territory_centroids_meta.json",
]


def get_r2_client():
    """Returns a boto3 S3 client configured for Cloudflare R2 endpoint if credentials are set."""
    if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        return None
    try:
        import boto3
        from botocore.config import Config

        endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
    except Exception as e:
        print(f"[R2 ERROR] Could not initialize R2 client: {e}")
        return None


def sync_from_r2(target_dir: str = "data", files: Optional[List[str]] = None) -> bool:
    """Download required data files from Cloudflare R2 bucket if not present locally."""
    client = get_r2_client()
    if not client:
        return False

    os.makedirs(target_dir, exist_ok=True)
    sync_list = files or DEFAULT_SYNC_FILES
    success_count = 0

    for filename in sync_list:
        target_path = os.path.join(target_dir, filename)
        if not os.path.exists(target_path):
            try:
                print(f"[R2] Syncing {filename} from bucket '{R2_BUCKET_NAME}'...")
                client.download_file(R2_BUCKET_NAME, filename, target_path)
                print(f"[R2] Downloaded {filename} ({os.path.getsize(target_path)} bytes).")
                success_count += 1
            except Exception as e:
                print(f"[R2 WARN] Could not download {filename} from R2: {e}")

    return success_count > 0


def upload_file_to_r2(local_path: str, remote_filename: Optional[str] = None) -> bool:
    """Upload a local data file to Cloudflare R2."""
    client = get_r2_client()
    if not client:
        print("[R2 ERROR] Missing R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, or R2_SECRET_ACCESS_KEY.")
        return False

    if not os.path.exists(local_path):
        print(f"[R2 ERROR] Local file not found: {local_path}")
        return False

    dest_key = remote_filename or os.path.basename(local_path)
    try:
        print(f"[R2] Uploading {local_path} to {R2_BUCKET_NAME}/{dest_key}...")
        client.upload_file(local_path, R2_BUCKET_NAME, dest_key)
        print(f"[R2] Successfully uploaded {dest_key}.")
        return True
    except Exception as e:
        print(f"[R2 ERROR] Failed to upload {dest_key}: {e}")
        return False

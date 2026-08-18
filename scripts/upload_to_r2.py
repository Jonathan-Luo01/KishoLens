#!/usr/bin/env python3
"""
scripts/upload_to_r2.py — Upload all local database, cache, and vector files to Cloudflare R2.

Usage:
  export R2_ACCOUNT_ID="your_account_id"
  export R2_ACCESS_KEY_ID="your_access_key_id"
  export R2_SECRET_ACCESS_KEY="your_secret_key"
  export R2_BUCKET_NAME="kisholens-data"
  python scripts/upload_to_r2.py
"""

import os
import sys
from kisholens.storage.r2 import upload_file_to_r2, DEFAULT_SYNC_FILES

def main():
    print("=" * 60)
    print("  KishoLens Cloudflare R2 Batch Uploader")
    print("=" * 60)

    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory '{data_dir}' not found.")
        sys.exit(1)

    uploaded = 0
    for filename in DEFAULT_SYNC_FILES:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"\nProcessing {filename} ({os.path.getsize(filepath) / (1024*1024):.2f} MB)...")
            success = upload_file_to_r2(filepath, filename)
            if success:
                uploaded += 1
        else:
            print(f"\n[SKIP] {filename} does not exist locally.")

    print("\n" + "=" * 60)
    print(f"Upload complete! {uploaded}/{len(DEFAULT_SYNC_FILES)} files synced to Cloudflare R2.")
    print("=" * 60)

if __name__ == "__main__":
    main()

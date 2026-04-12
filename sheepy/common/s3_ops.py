"""Utilities for downloading resources from S3-compatible storage."""
from __future__ import annotations

from pathlib import Path

import requests


def download_resource(key: str, output_path: str | Path) -> Path:
    """Download a file from the sheepy S3 bucket.

    Args:
        key: Object key in the bucket.
        output_path: Local path to write the file.

    Returns:
        The output path as a ``Path`` object.
    """
    output_path = Path(output_path)
    url = f"https://robmsylvester.s3.us-west-1.amazonaws.com/{key}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return output_path

"""Timestamp utilities."""
from __future__ import annotations

from datetime import datetime, timezone


def get_timestamp_now() -> str:
    """UTC timestamp string suitable for filesystem-safe names (``YYYY_MM_DD__HH_MM_TZ``)."""
    now = datetime.now(tz=timezone.utc)
    return now.strftime("%Y_%m_%d__%H_%M_%Z")

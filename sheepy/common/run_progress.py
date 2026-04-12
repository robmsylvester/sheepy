"""High-visibility console progress lines for scripts and demos.

Uses stdout with flush so messages appear immediately, even when other
libraries buffer their own output.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime


def _ts() -> str:
    """Current local time formatted for progress-line prefixes."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_banner(title: str, width: int = 72) -> None:
    """Print a framed section title."""
    line = "=" * width
    print(f"\n{line}", flush=True)
    print(f"  [{_ts()}]  {title}", flush=True)
    print(f"{line}\n", flush=True)


def run_phase_start(phase: str, detail: str = "") -> None:
    """Mark the start of a named phase (e.g. training, data load)."""
    extra = f" — {detail}" if detail else ""
    print(f"[{_ts()}]  >>  START  {phase}{extra}", flush=True)


def run_phase_end(phase: str, detail: str = "") -> None:
    """Mark the end of a named phase."""
    extra = f" — {detail}" if detail else ""
    print(f"[{_ts()}]  <<  DONE   {phase}{extra}", flush=True)


def run_note(message: str) -> None:
    """A single progress note between start/end."""
    print(f"[{_ts()}]      ·  {message}", flush=True)


def run_context() -> None:
    """Log process id and cwd once at script entry (optional)."""
    print(
        f"[{_ts()}]  Process pid={os.getpid()}  cwd={os.getcwd()}",
        flush=True,
    )


def configure_demo_logging() -> None:
    """Force line-buffered stdout so progress lines interleave with other prints."""
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

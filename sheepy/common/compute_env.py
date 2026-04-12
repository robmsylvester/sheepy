"""Resolve Lightning ``accelerator`` / ``devices`` from hardware and optional config.

PyPI ``torch`` wheels are often CPU-only; ``torch.cuda.is_available()`` is only
``True`` when a CUDA-enabled build is installed (see project README).
"""
from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "resolve_trainer_accelerator_and_devices",
    "strategy_for_distributed",
    "summarize_compute_environment",
]


def summarize_compute_environment() -> dict[str, Any]:
    """Human-readable facts for logging (W&B config, console)."""
    cuda = torch.cuda.is_available()
    n_cuda = torch.cuda.device_count() if cuda else 0
    names: list[str] = []
    if cuda:
        for i in range(n_cuda):
            try:
                names.append(torch.cuda.get_device_name(i))
            except Exception:
                names.append(f"device_{i}")
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return {
        "torch_cuda_available": cuda,
        "torch_cuda_device_count": n_cuda,
        "torch_cuda_device_names": names,
        "torch_mps_available": mps,
        "torch_version": torch.__version__,
    }


def resolve_trainer_accelerator_and_devices(config: dict[str, Any]) -> tuple[str, int | str]:
    """Pick Lightning ``Trainer(accelerator=..., devices=...)``.

    Priority:
    1. Explicit ``config["accelerator"]`` (and optional ``config["devices"]``).
    2. NVIDIA CUDA when ``torch.cuda.is_available()`` — default ``devices=1`` (first GPU).
    3. Apple MPS when available.
    4. CPU otherwise.

    Set ``hparams.devices`` to ``"auto"`` or an integer to use multiple GPUs (DDP).
    """
    user_acc = config.get("accelerator")
    user_dev = config.get("devices")

    if user_acc is not None and str(user_acc).strip():
        acc = str(user_acc).strip().lower()
        # Lightning accepts "gpu" as legacy alias; normalize for logs only — Trainer accepts both.
        dev: int | str = user_dev if user_dev is not None else "auto"
        return acc, dev

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if user_dev is not None:
            return "cuda", user_dev
        # Single-GPU default avoids surprising multi-GPU DDP on shared workstations.
        return "cuda", 1 if n >= 1 else "auto"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", user_dev if user_dev is not None else 1

    return "cpu", user_dev if user_dev is not None else "auto"


def strategy_for_distributed(accelerator: str, devices: int | str) -> str:
    """Return ``ddp`` only when multiple CUDA processes are requested."""
    num_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if accelerator == "cuda" and (
        (isinstance(devices, int) and devices > 1)
        or (devices == "auto" and num_cuda > 1)
    ):
        return "ddp"
    return "auto"

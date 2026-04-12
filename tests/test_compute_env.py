"""Tests for sheepy.common.compute_env."""
from unittest.mock import patch

import pytest

from sheepy.common.compute_env import (
    resolve_trainer_accelerator_and_devices,
    strategy_for_distributed,
    summarize_compute_environment,
)


def test_summarize_contains_torch_keys():
    s = summarize_compute_environment()
    assert "torch_version" in s
    assert "torch_cuda_available" in s
    assert "torch_mps_available" in s


def test_user_accelerator_override():
    acc, dev = resolve_trainer_accelerator_and_devices({"accelerator": "cpu", "devices": 1})
    assert acc == "cpu"
    assert dev == 1


@patch("sheepy.common.compute_env.torch.cuda.is_available", return_value=True)
@patch("sheepy.common.compute_env.torch.cuda.device_count", return_value=2)
def test_cuda_defaults_to_one_device(mock_count, mock_avail):
    acc, dev = resolve_trainer_accelerator_and_devices({})
    assert acc == "cuda"
    assert dev == 1


@patch("sheepy.common.compute_env.torch.cuda.is_available", return_value=True)
@patch("sheepy.common.compute_env.torch.cuda.device_count", return_value=2)
def test_cuda_multi_when_requested(mock_count, mock_avail):
    acc, dev = resolve_trainer_accelerator_and_devices({"devices": 2})
    assert acc == "cuda"
    assert dev == 2


@patch("sheepy.common.compute_env.torch.cuda.is_available", return_value=True)
@patch("sheepy.common.compute_env.torch.cuda.device_count", return_value=2)
def test_strategy_ddp_for_multi(mock_count, mock_avail):
    assert strategy_for_distributed("cuda", 2) == "ddp"
    assert strategy_for_distributed("cuda", 1) == "auto"
    assert strategy_for_distributed("cuda", "auto") == "ddp"


@patch("sheepy.common.compute_env.torch.cuda.is_available", return_value=True)
@patch("sheepy.common.compute_env.torch.cuda.device_count", return_value=1)
def test_strategy_single_gpu_auto_devices(mock_count, mock_avail):
    assert strategy_for_distributed("cuda", "auto") == "auto"


@patch("sheepy.common.compute_env.torch.cuda.is_available", return_value=False)
@patch("sheepy.common.compute_env.torch.backends.mps.is_available", return_value=False)
def test_cpu_when_no_cuda(mock_mps, mock_cuda):
    acc, dev = resolve_trainer_accelerator_and_devices({})
    assert acc == "cpu"

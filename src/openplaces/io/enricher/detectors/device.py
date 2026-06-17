"""Torch device selection shared by enrichment detectors."""

from __future__ import annotations

import os
from functools import cache


@cache
def get_device(force: str | None = None):
    """Select the best available inference device: cuda, DirectML, or CPU.

    The choice is printed once per process so a silent CPU fallback is
    visible instead of looking like a hang.

    Parameters
    ----------
    force : str, optional
        Device string (e.g. 'cpu', 'cuda:0') that overrides detection.
        The OPENPLACES_DEVICE environment variable plays the same role
        when set.

    Returns
    -------
    torch.device
        CUDA when an NVIDIA GPU is available (ROCm builds also surface
        here), otherwise DirectML when the optional torch_directml
        package is importable (AMD/Intel GPUs on Windows; never install
        it into the main conda env), otherwise CPU.
    """
    import torch

    force = force or os.environ.get('OPENPLACES_DEVICE')
    if force:
        device = torch.device(force)
        print(f'Inference device: {device} (forced)')
        return device

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f'Inference device: {device} ({torch.cuda.get_device_name(0)})')
        return device

    try:
        import torch_directml
    except ImportError:
        pass
    else:
        device = torch_directml.device()
        name = torch_directml.device_name(0)
        print(f'Inference device: {device} (DirectML, {name})')
        return device

    print('Inference device: cpu (no GPU backend available)')
    return torch.device('cpu')

"""Download and cache pretrained enrichment models."""

from __future__ import annotations

import tempfile
from pathlib import Path

from openplaces.config import cfg


def get_model(
    url: str,
    filename: str,
    model_path: str | Path | None = None,
    label: str = 'model',
) -> Path:
    """Return a local model path, downloading the model when needed."""
    destination = (
        Path(model_path)
        if model_path is not None
        else Path(cfg.models_dir) / 'external' / 'brails' / filename
    )
    if destination.exists():
        print(f'{label.capitalize()} loaded from {destination}')
        return destination

    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {label} to {destination} ...')
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f'.{destination.name}.',
            suffix='.tmp',
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    temp_file.write(chunk)
        temp_path.replace(destination)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    print(f'{label.capitalize()} downloaded.')
    return destination

"""
Tombstone receipts and the per-run context they record: the
relative-path convention every receipt and footer uses, whether
a run is orchestrated, and the write, read and discard of a
receipt beside an output's would-be path.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from openplaces.config import cfg

RECEIPT_SUFFIX = '.consumed.json'

_RECEIPT_FORMAT = 1


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def _relative_posix(path) -> str:
    """Data-root-relative forward-slash path (portable across OSes)."""
    path = Path(path)
    try:
        return path.relative_to(cfg.data_root).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_relative(path_str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else cfg.data_root / p


def is_orchestrated() -> bool:
    """True when running under an orchestrator (e.g. Snakemake).

    Orchestrated runs must produce the physical output file, so
    receipt-based skips are voided.
    """
    return bool(
        os.environ.get('SNAKEMAKE') or os.environ.get('OPENPLACES_ORCHESTRATED')
    )


def _cleanup_config() -> dict:
    return (cfg.get('retention') or {}).get('cleanup') or {}


def _recipe_retention_override(recipe_id) -> str | None:
    """Per-recipe retention set in the user's config, if any.

    Mirrors the retention.recipes lookup in
    :meth:`~openplaces.config.OpenPlacesConfig.retention_for`.
    """
    if recipe_id is None:
        return None
    recipes = (cfg.get('retention') or {}).get('recipes') or {}
    return recipes.get(str(recipe_id))


# RECEIPTS


def receipt_path(output_path) -> Path:
    """Path of the tombstone receipt for an output file or directory."""
    p = Path(output_path)
    return p.with_name(p.stem + RECEIPT_SUFFIX)


def write_receipt(output_path, receipt: dict) -> Path:
    """Atomically write a tombstone receipt beside an output path."""
    receipt = {'format': _RECEIPT_FORMAT, **receipt}
    rp = receipt_path(output_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    tmp = rp.with_name(rp.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    os.replace(tmp, rp)
    return rp


def read_receipt(output_path) -> dict | None:
    """Return the receipt dict for an output path, or None if absent/corrupt."""
    rp = receipt_path(output_path)
    if not rp.exists():
        return None
    try:
        return json.loads(rp.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def discard_receipt(output_path) -> None:
    """Remove the receipt for an output path (tolerant of it being absent)."""
    receipt_path(output_path).unlink(missing_ok=True)

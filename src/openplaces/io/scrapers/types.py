"""
Lightweight stand-ins for the BRAILS++ AssetInventory / ImageSet types.

These replace the corresponding brails.types classes so the scrapers can run
without the full BRAILS++ package installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace


@dataclass
class AssetInventory:
    """Minimal inventory of spatial assets with footprint coordinates.

    Parameters
    ----------
    inventory
        Mapping from asset ID to an object with a ``.coordinates`` attribute
        containing a list of ``(lon, lat)`` tuples that trace the footprint.
    """

    inventory: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, coords_by_id: dict[object, list]) -> AssetInventory:
        """Build an inventory from a plain ``{id: [(lon, lat), ...]}`` dict."""
        inventory = {
            key: SimpleNamespace(coordinates=coords)
            for key, coords in coords_by_id.items()
        }
        return cls(inventory=inventory)


@dataclass
class Image:
    """Single image entry stored in an `ImageSet`.

    Parameters
    ----------
    filename
        Basename of the image file (no directory prefix).
    metadata
        Optional per-image metadata dictionary (e.g. camera parameters).
    """

    filename: str
    metadata: dict = field(default_factory=dict)


class ImageSet:
    """Collection of downloaded images keyed by asset ID.

    Attributes
    ----------
    dir_path
        Root directory that contains the image files.
    images
        Mapping from asset ID to `Image`.
    counts
        Tally of how the images were obtained (e.g. ``cached``,
        ``downloaded``), set by the scraper that produced the set.
    """

    def __init__(self) -> None:
        self.dir_path: str = ''
        self.images: dict = {}
        self.counts: dict = {}

    def add_image(self, key: object, image: Image) -> None:
        self.images[key] = image

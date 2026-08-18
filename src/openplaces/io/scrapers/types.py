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
    """Single image held in an `ImageSet`.

    Imagery is never persisted by openplaces: Google's Static API policy
    prohibits "pre-fetching, indexing, storing, or caching" of content, so an
    image exists only for the duration of the inference call that consumes
    it. `payload` carries the encoded bytes in memory; `path` is a fallback
    for scrapers that can only write to a scratch directory, which the caller
    deletes before returning.

    Parameters
    ----------
    filename
        Basename used for logging and metadata only.
    payload
        Encoded image bytes held in memory, when the scraper can supply them.
    path
        Location in a caller-owned temporary directory, for scrapers that
        cannot yet return bytes. Never a durable location.
    metadata
        Optional per-image metadata (e.g. camera parameters, panorama id).
        Panorama and place ids are the one thing the policy permits storing
        indefinitely, so these are safe to persist as provenance.
    """

    filename: str
    payload: bytes | None = None
    path: str | None = None
    metadata: dict = field(default_factory=dict)

    def open(self):
        """Return an open PIL image, from memory when possible."""
        from io import BytesIO

        from PIL import Image as PILImage

        if self.payload is not None:
            return PILImage.open(BytesIO(self.payload))
        if self.path is not None:
            return PILImage.open(self.path)
        raise ValueError(f'Image {self.filename!r} carries no payload or path.')


class ImageSet:
    """Collection of fetched images keyed by asset ID.

    There is deliberately no directory attribute: nothing here is stored, so
    there is no location to record. See `Image` for why.

    Attributes
    ----------
    images
        Mapping from asset ID to `Image`.
    counts
        Tally of how the images were obtained (e.g. ``fetched``, ``failed``),
        set by the scraper that produced the set.
    """

    def __init__(self) -> None:
        self.images: dict = {}
        self.counts: dict = {}

    def add_image(self, key: object, image: Image) -> None:
        self.images[key] = image

"""Single-attribute estimators of the curate stage: each module fills
or classifies one attribute from the evidence on the row, marking what
it wrote. Imported by the curator's step loader, which otherwise skips
sub-packages, so the steps here register like any top-level module's.
"""

from openplaces.io.curator.estimators import (  # noqa: F401
    manufactured_homes,
    occupancy,
    stories,
)

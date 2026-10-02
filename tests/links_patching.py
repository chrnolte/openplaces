"""Patch a name on every `io.harmonizer.links` submodule that binds it.

Until 2026-09-30 the `links` package forwarded attribute assignments to
its submodules (a class swap in its package file), so that the tests
written against the single module it used to be kept working. That
forwarding is gone; a test that patched `links.get_entities` now says
which modules it means through this helper, which sets the name on each
submodule whose namespace holds it, the same set the forwarder reached.
A step looks names up in its own module's globals (AGENTS.md, module
layer hierarchy), so a patch on the package alone would reach nothing.
"""

from __future__ import annotations

from openplaces.io.harmonizer import links

SUBMODULES = (
    links._shared,
    links.spatial,
    links.sidecars,
    links.points,
    links.discovery,
    links.combine,
    links.by_id,
    links.address_ranges,
    links.additions,
    links.condo_clusters,
    links.overlaps,
)


def patch_links(monkeypatch, name: str, value) -> None:
    """Set *name* to *value* on every links submodule that defines it.

    Raises if no submodule binds the name, so a patch that reaches
    nothing fails loudly instead of passing by accident.
    """
    holders = [module for module in SUBMODULES if name in vars(module)]
    if not holders:
        raise AttributeError(f'no links submodule binds {name!r}')
    for module in holders:
        monkeypatch.setattr(module, name, value)

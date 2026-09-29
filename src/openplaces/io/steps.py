"""A stage's named pipeline steps: the registry decorator and the lazy
loader.

The harmonize, enrich and curate packages each dispatch recipe
``pipeline`` entries by name to a function registered under that name.
Until 2026-09-22 each package carried its own copy of the decorator and
of the loader that imports the step submodules; the copies differed only
in that the harmonizer's decorator also records a phase. This module is
the one copy. It sits below the stages in the layer hierarchy and
imports nothing from openplaces, so any stage may use it.

Each stage keeps its registry as a module-level dict of its own
(``_STEP_REGISTRY``, and the harmonizer's ``_STEP_PHASES``): tests
monkeypatch those dicts by name, and ``harmonizer.links`` reads the
phases directly, so the dicts stay where they were and only the code
that fills them is shared.
"""

from __future__ import annotations

import pkgutil
from collections.abc import Callable
from importlib import import_module

#: The two phases a harmonize step can declare. Geometry-phase steps
#: mutate spine rows or geometry or run spatial joins, and their configs
#: are part of the link-sidecar fingerprint; attribute-phase steps only
#: read or annotate.
PHASES = ('geometry', 'attributes')


def make_register(
    registry: dict[str, Callable],
    phases: dict[str, str] | None = None,
    vocabulary: tuple[str, ...] = PHASES,
    default_phase: str | None = 'attributes',
) -> Callable:
    """Return a decorator that registers a step under one or more names.

    Parameters
    ----------
    registry : dict
        The stage's own name-to-function mapping, filled in place.
    phases : dict, optional
        Where given, the returned decorator also accepts ``phase`` and
        records it here per name. Without it the decorator refuses a
        phase, so a stage that has no phases cannot be handed one by
        mistake.
    vocabulary : tuple of str
        The phases this stage's steps may declare. The harmonizer's
        two (geometry, attributes) are the default; the curator passes
        its own five, in pipeline order.
    default_phase : str or None
        The phase a step gets when it declares none. None makes the
        phase mandatory, which is what a stage whose recipes are
        checked for phase order wants: an untagged step there is a
        step nobody has placed.

    Returns
    -------
    Callable
        ``_register(*names, phase=...)``, used as ``@_register('name')``.
    """

    def _register(*names: str, phase: str | None = None) -> Callable:
        if phases is None:
            if phase is not None:
                raise TypeError('this stage has no step phases')
        else:
            if phase is None:
                if default_phase is None:
                    raise TypeError(
                        f'{names[0]!r}: this stage requires a phase, one of '
                        f'{", ".join(vocabulary)}'
                    )
                phase = default_phase
            if phase not in vocabulary:
                raise ValueError(
                    f'phase must be one of {", ".join(vocabulary)}, got {phase!r}'
                )

        def decorator(fn: Callable) -> Callable:
            for name in names:
                registry[name] = fn
                if phases is not None:
                    phases[name] = phase
            return fn

        return decorator

    return _register


def make_loader(package: str, path) -> Callable[[], None]:
    """Return a function that imports a package's step submodules once.

    The import is deferred until a step is first dispatched, so merely
    importing a stage package (to use its state dataclass in a test, say)
    does not pull every step module. Subpackages are skipped, which keeps
    the enricher's heavy ``detectors`` out of the import.

    Parameters
    ----------
    package : str
        The stage package's ``__name__``.
    path
        The stage package's ``__path__``.
    """
    loaded = False

    def _load_steps() -> None:
        nonlocal loaded
        if loaded:
            return
        for module in pkgutil.iter_modules(path):
            if not module.ispkg:
                import_module(f'{package}.{module.name}')
        loaded = True

    return _load_steps

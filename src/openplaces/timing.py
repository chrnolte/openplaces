"""
openplaces timing module

Track execution time across script milestones and export as structured data.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

from openplaces.path import logs_path

__all__ = ['Timer', 'get_timer', 'log_step']


@dataclass
class TimerRecord:
    """Single timing record."""

    label: str
    start: float
    end: float | None = None
    cpu_start: float = 0.0
    cpu_end: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        if self.end is None:
            return None
        return self.end - self.start

    @property
    def cpu_duration(self) -> float | None:
        if self.cpu_end is None:
            return None
        return self.cpu_end - self.cpu_start

    def to_dict(self) -> dict[str, Any]:
        result = {
            'label': self.label,
            'duration_seconds': self.duration,
            'cpu_seconds': self.cpu_duration,
        }
        if self.metadata:
            result['metadata'] = self.metadata
        return result


@dataclass
class Timer:
    """
    Track time consumption across labeled milestones.

    Two modes of use:
    1. Milestones: mark points in time, duration = time since last mark
    2. Steps: explicit begin/end or context manager for nested sections

    Example
    -------
    >>> timer = Timer('process_county')
    >>> timer.mark('load')
    >>> data = load_data()
    >>> timer.mark('transform')
    >>> result = transform(data)
    >>> timer.mark('export')
    >>> export(result)
    >>> timer.finish()
    >>> timer.save('timings.json')
    """

    name: str
    admin_id: str | None = None
    logger: logging.Logger | None = None
    _created: float = field(default_factory=perf_counter)
    _cpu_created: float = field(default_factory=process_time)
    _last_mark: float = field(default=None, repr=False)
    _last_cpu_mark: float = field(default=None, repr=False)
    _records: list[TimerRecord] = field(default_factory=list, repr=False)
    _active_step: TimerRecord | None = field(default=None, repr=False)
    _finished: bool = field(default=False, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._last_mark = self._created
        self._last_cpu_mark = self._cpu_created

    def mark(self, label: str, **metadata) -> tuple[float, float]:
        """Record a milestone. Duration = time since last mark (or creation)."""
        now = perf_counter()
        cpu_now = process_time()
        record = TimerRecord(
            label=label,
            start=self._last_mark,
            end=now,
            cpu_start=self._last_cpu_mark,
            cpu_end=cpu_now,
            metadata=metadata,
        )
        self._records.append(record)
        self._last_mark = now
        self._last_cpu_mark = cpu_now
        if self.logger:
            self.logger.info(
                f'{label:.<62} {record.duration:>8.2f}s  '
                f'{record.cpu_duration:>8.2f}s cpu'
            )
        return record.duration, record.cpu_duration

    def begin(self, label: str, **metadata):
        """Begin a timed step (for explicit begin/end usage)."""
        if self._active_step is not None:
            self.end()
        self._active_step = TimerRecord(
            label=label,
            start=perf_counter(),
            cpu_start=process_time(),
            metadata=metadata,
        )

    def end(self) -> tuple[float, float] | None:
        """End the current timed step. Returns (wall_duration, cpu_duration)."""
        if self._active_step is None:
            return None
        self._active_step.end = perf_counter()
        self._active_step.cpu_end = process_time()
        self._records.append(self._active_step)
        duration = (self._active_step.duration, self._active_step.cpu_duration)
        # Update last mark so subsequent mark() calls are consistent
        self._last_mark = self._active_step.end
        self._last_cpu_mark = self._active_step.cpu_end
        if self.logger:
            self.logger.info(
                f'{self._active_step.label:.<62} {duration[0]:>8.2f}s  '
                f'{duration[1]:>8.2f}s cpu'
            )
        self._active_step = None
        return duration

    def finish(self, label: str = '_final'):
        if self._finished:
            return
        if self._active_step is not None:
            self.end()
        now = perf_counter()
        cpu_now = process_time()
        if now - self._last_mark > 0.001:
            record = TimerRecord(
                label=label,
                start=self._last_mark,
                end=now,
                cpu_start=self._last_cpu_mark,
                cpu_end=cpu_now,
            )
            self._records.append(record)
        self._finished = True

    @property
    def total_duration(self) -> float:
        """Total elapsed time since timer creation."""
        return perf_counter() - self._created

    @property
    def tracked_duration(self) -> float:
        """Sum of all recorded step durations."""
        return sum(r.duration for r in self._records if r.duration is not None)

    def _serialize_value(self, value: Any) -> Any:
        """Convert non-JSON-serializable values."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        return value

    @property
    def records(self) -> list[dict[str, Any]]:
        """All timing records as dictionaries."""
        records = []
        for r in self._records:
            rec_dict = r.to_dict()
            if 'metadata' in rec_dict:
                rec_dict['metadata'] = self._serialize_value(rec_dict['metadata'])
            records.append(rec_dict)
        return records

    def to_dict(self) -> dict[str, Any]:
        """Export full timing data as dictionary."""
        return {
            'name': self.name,
            'admin_id': self.admin_id,
            'total_seconds': self.total_duration,
            'tracked_seconds': self.tracked_duration,
            'metadata': self._serialize_value(self.metadata),
            'steps': self.records,
        }

    def _get_default_log_path(self, extension: str = 'json') -> Path:
        """Generate default log path using openplaces config."""
        from datetime import datetime

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'{self.name}_{timestamp}.{extension}'
        return logs_path(self.admin_id, filename=filename)

    def save(self, path: Path | str | None = None):
        """Save timing data to JSON file. Auto-finishes if needed."""
        if not self._finished:
            self.finish()

        if path is None:
            path = self._get_default_log_path('json')
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, path: Path | str | None = None):
        """Save timing data to CSV (one row per step). Auto-finishes if needed."""
        if not self._finished:
            self.finish()
        import csv

        if path is None:
            path = self._get_default_log_path('csv')
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['label', 'duration_seconds', 'cpu_seconds']
            )
            writer.writeheader()
            writer.writerows(self.records)

    def summary(self) -> None:
        """Human-readable summary."""
        if not self._finished:
            self.finish()
        lines = [
            f'Timer: {self.name}',
            f'Total: {self.total_duration:.2f}s',
            '',
        ]
        for r in self._records:
            if r.duration is not None:
                pct = (
                    100 * r.duration / self.total_duration
                    if self.total_duration > 0
                    else 0
                )
                lines.append(
                    f'  {r.label:.<58} {r.duration:>8.2f}s  '
                    f'{r.cpu_duration:>8.2f}s cpu ({pct:>5.1f}%)'
                )
        print('\n'.join(lines))

    @property
    def verbose(self) -> bool:
        return self.logger is not None and bool(self.logger.handlers)


# Global timer registry
_timers: dict[str, Timer] = {}


def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f'openplaces.timing.{name}')
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter('\033[30;48;2;245;245;255m%(message)s\033[0m')
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_timer(
    name: str = 'default',
    admin_id: str | None = None,
    logger: logging.Logger | None = None,
    verbose: bool = False,
    overwrite: bool = False,
    **metadata,
) -> Timer:
    if name not in _timers or overwrite:
        if logger is None and verbose:
            logger = _make_logger(name)
        timer = Timer(name=name, admin_id=admin_id, logger=logger)
        timer.metadata.update(metadata)
        _timers[name] = timer
    else:
        existing = _timers[name]
        if verbose and not existing.verbose:
            existing.logger = _make_logger(name)
        elif not verbose and existing.verbose:
            existing.logger = None
    return _timers[name]


def clear_timers():
    """Clear all registered timers."""
    _timers.clear()


@contextmanager
def log_step(
    label: str,
    timer: Timer | str | None = None,
    **metadata,
):
    if isinstance(timer, str):
        timer = get_timer(timer)

    if timer:
        timer.begin(label, **metadata)

    try:
        yield
    finally:
        if timer:
            timer.end()

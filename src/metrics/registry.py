"""Deterministic registry for available metrics."""

from __future__ import annotations

from collections.abc import Iterator

from .base import Metric


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        name = getattr(metric, "name", None)
        stage = getattr(metric, "stage", None)
        if not isinstance(name, str) or not name:
            raise ValueError("metric must define a non-empty name")
        if not isinstance(stage, str) or not stage:
            raise ValueError("metric must define a non-empty stage")
        if name in self._metrics:
            raise ValueError(f"metric {name!r} is already registered")
        self._metrics[name] = metric

    def get(self, name: str) -> Metric:
        try:
            return self._metrics[name]
        except KeyError as exc:
            raise KeyError(f"unknown metric: {name}") from exc

    def all(self) -> list[Metric]:
        return list(self._metrics.values())

    def __iter__(self) -> Iterator[Metric]:
        return iter(self._metrics.values())

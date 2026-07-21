"""Shared metric result and implementation contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Protocol


VALID_STATUSES = {"ok", "skipped", "error"}


@dataclass
class MetricResult:
    """Serializable result returned by every metric."""

    trace_id: str
    metric_name: str
    stage: str
    score: float | None
    label: str
    status: str = "ok"
    evidence: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    runtime_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1 or null")
        if self.runtime_ms < 0:
            raise ValueError("runtime_ms must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


class Metric(Protocol):
    """Protocol implemented by each concrete metric."""

    name: str
    stage: str

    def compute(self, trace: dict[str, Any], config: dict[str, Any]) -> MetricResult:
        ...

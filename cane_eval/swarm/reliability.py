"""
reliability.py -- Swarm reliability scoring.

Extends the 3-pillar single-agent model to a 4-pillar swarm model:
  Outcome (final answer quality) + Process (orchestration quality)
  + Structural (schema) + Performance (latency)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SwarmReliabilityConfig:
    """
    Weight configuration for swarm reliability scoring.

    Four pillars with configurable weights that must sum to 1.0.
    When a pillar has no data, its weight is redistributed proportionally.
    """
    outcome_weight: float = 0.35
    process_weight: float = 0.35
    structural_weight: float = 0.15
    performance_weight: float = 0.15

    def __post_init__(self):
        total = (
            self.outcome_weight
            + self.process_weight
            + self.structural_weight
            + self.performance_weight
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Swarm reliability weights must sum to 1.0, got {total:.2f} "
                f"({self.outcome_weight}:{self.process_weight}:"
                f"{self.structural_weight}:{self.performance_weight})"
            )


def compute_swarm_reliability(
    outcome_score: float,
    process_score: float,
    schema_score: Optional[float] = None,
    latency_score: Optional[float] = None,
    config: Optional[SwarmReliabilityConfig] = None,
) -> tuple[float, str]:
    """
    Compute Swarm Reliability Score.

    Four pillars:
    - Outcome (final answer quality from judge, 0-100)
    - Process (orchestration quality: routing + handoff + context + convergence, 0-100)
    - Structural (schema adherence, 0-100, optional)
    - Performance (latency vs target, 0-100, optional)

    When structural or performance data is unavailable, weights are
    redistributed proportionally among the remaining pillars.

    Args:
        outcome_score: Final answer quality (0-100)
        process_score: Orchestration quality (0-100)
        schema_score: Schema pass percentage (0-100), or None
        latency_score: Latency score (0-100), or None
        config: Weight configuration

    Returns:
        (score, grade) tuple where score is 0-100 and grade is A-F.
    """
    if config is None:
        config = SwarmReliabilityConfig()

    # Build available pillars
    pillars = {
        "outcome": (outcome_score, config.outcome_weight),
        "process": (process_score, config.process_weight),
    }

    if schema_score is not None:
        pillars["structural"] = (schema_score, config.structural_weight)

    if latency_score is not None:
        pillars["performance"] = (latency_score, config.performance_weight)

    # Redistribute missing weights proportionally
    total_available_weight = sum(w for _, w in pillars.values())
    if total_available_weight <= 0:
        return 0.0, "F"

    reliability = sum(
        score * (weight / total_available_weight)
        for score, weight in pillars.values()
    )

    reliability = round(reliability, 1)

    # Grade (same scale as single-agent)
    if reliability >= 90:
        grade = "A"
    elif reliability >= 75:
        grade = "B"
    elif reliability >= 60:
        grade = "C"
    elif reliability >= 40:
        grade = "D"
    else:
        grade = "F"

    return reliability, grade

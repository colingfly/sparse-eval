"""
sparse_eval.swarm -- Multi-agent orchestration evaluation.

Evaluate how well agents work together: routing, handoffs,
context preservation, convergence, and final answer quality.
"""

from sparse_eval.swarm.models import (
    SwarmTopologyType,
    SwarmMessageRole,
    SwarmAgent,
    SwarmTopology,
    SwarmMessage,
    SwarmTrace,
    SwarmTestCase,
    SwarmResult,
    SwarmSummary,
)
from sparse_eval.swarm.suite import SwarmSuite
from sparse_eval.swarm.trace import parse_trace, load_trace_file
from sparse_eval.swarm.judge import SwarmJudge
from sparse_eval.swarm.engine import SwarmReliabilityRunner
from sparse_eval.swarm.runner import SwarmRunner
from sparse_eval.swarm.reliability import SwarmReliabilityConfig, compute_swarm_reliability
from sparse_eval.swarm.criteria import (
    SwarmEvalContext,
    RoutingCriterion,
    HandoffCriterion,
    ContextPreservationCriterion,
    ConvergenceCriterion,
    CollaborationCriterion,
)

__all__ = [
    # Models
    "SwarmTopologyType",
    "SwarmMessageRole",
    "SwarmAgent",
    "SwarmTopology",
    "SwarmMessage",
    "SwarmTrace",
    "SwarmTestCase",
    "SwarmResult",
    "SwarmSummary",
    # Suite
    "SwarmSuite",
    # Trace
    "parse_trace",
    "load_trace_file",
    # Judge
    "SwarmJudge",
    # Engine
    "SwarmReliabilityRunner",
    # Runner
    "SwarmRunner",
    # Reliability
    "SwarmReliabilityConfig",
    "compute_swarm_reliability",
    # Criteria
    "SwarmEvalContext",
    "RoutingCriterion",
    "HandoffCriterion",
    "ContextPreservationCriterion",
    "ConvergenceCriterion",
    "CollaborationCriterion",
]

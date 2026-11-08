"""
sparse-eval -- AI system reliability infrastructure.

Evaluate any AI system's reliability across correctness, structure, and performance.
Extensible criteria, configurable weights, parallel execution, and domain integrations.
"""

__version__ = "1.0.0"

from sparse_eval.providers import get_provider, AnthropicProvider, OpenAIProvider, GeminiProvider, OpenAICompatibleProvider
from sparse_eval.suite import ReliabilitySuite, ReliabilityCase, TestSuite, TestCase
from sparse_eval.judge import Judge, JudgeResult, CriteriaScore
from sparse_eval.engine import (
    ReliabilityRunner, ReliabilityResult, ReliabilitySummary,
    EvalRunner, EvalResult, RunSummary,
    LatencyStats, SchemaResult,
)
from sparse_eval.export import Exporter
from sparse_eval.mining import FailureMiner
from sparse_eval.rca import RootCauseAnalyzer, RCAResult, TargetedRCAResult
from sparse_eval.criteria import CriterionPlugin, CriteriaRegistry, EvalContext, CriterionType
from sparse_eval.reliability import ReliabilityConfig

# Integrations (lazy-loaded to avoid import errors if frameworks not installed)
from sparse_eval.integrations import (
    evaluate_langchain,
    evaluate_llamaindex,
    evaluate_openai,
    evaluate_fastapi,
    evaluate_phlote,
)

# Swarm (multi-agent evaluation)
from sparse_eval.swarm import (
    SwarmSuite,
    SwarmReliabilityRunner,
    SwarmJudge,
    SwarmResult,
    SwarmSummary,
    SwarmTrace,
    SwarmReliabilityConfig,
)

__all__ = [
    # Providers
    "get_provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "OpenAICompatibleProvider",
    # Core (new primary names)
    "ReliabilityRunner",
    "ReliabilitySuite",
    "ReliabilityCase",
    "ReliabilityResult",
    "ReliabilitySummary",
    # Core (deprecated aliases, backward compat)
    "EvalRunner",
    "TestSuite",
    "TestCase",
    "EvalResult",
    "RunSummary",
    # Criteria system
    "CriterionPlugin",
    "CriteriaRegistry",
    "EvalContext",
    "CriterionType",
    # Config
    "ReliabilityConfig",
    # Judge
    "Judge",
    "JudgeResult",
    "CriteriaScore",
    # Other
    "LatencyStats",
    "SchemaResult",
    "Exporter",
    "FailureMiner",
    "RootCauseAnalyzer",
    "RCAResult",
    "TargetedRCAResult",
    # Integrations
    "evaluate_langchain",
    "evaluate_llamaindex",
    "evaluate_openai",
    "evaluate_fastapi",
    "evaluate_phlote",
    # Swarm
    "SwarmSuite",
    "SwarmReliabilityRunner",
    "SwarmJudge",
    "SwarmResult",
    "SwarmSummary",
    "SwarmTrace",
    "SwarmReliabilityConfig",
]
